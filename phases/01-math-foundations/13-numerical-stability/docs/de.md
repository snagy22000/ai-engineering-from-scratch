# Numerische Stabilität

> Gleitkomma ist eine undichte Abstraktion. Beim Training trifft es dich – und du siehst es nicht kommen.

**Typ:** Aufbauen
**Sprachen:** Python
**Voraussetzungen:** Phase 1, Lektionen 01–04
**Zeit:** ~120 Minuten

## Lernziele

- Numerisch stabiles Softmax und Log-Sum-Exp mit dem Max-Subtraktions-Trick implementieren
- Overflow, Underflow und katastrophale Auslöschung in Gleitkomma-Berechnungen erkennen
- Analytische Gradienten über zentrierte finite Differenzen gegen numerische Gradienten verifizieren
- Erklären, warum bfloat16 fürs Training oft besser ist als float16 und wie Loss Scaling Gradient-Underflow verhindert

## Das Problem

Dein Modell trainiert drei Stunden, dann wird der Loss NaN. Du baust ein Print ein. Die Logits sind bei Schritt 9.000 unauffällig. Bei Schritt 9.001 sind sie `inf`. Bei Schritt 9.002 ist jeder Gradient `nan` und das Training ist tot.

Oder: Dein Modell trainiert bis zum Ende, aber die Genauigkeit liegt 2 % unter dem Paper. Du prüfst alles. Architektur passt. Hyperparameter passen. Daten passen. Das Problem: Das Paper nutzte float32, du nutztest float16 ohne korrektes Scaling. Zweiunddreißig Bit aufaddierter Rundungsfehler haben leise deine Genauigkeit aufgefressen.

Oder: Du implementierst Cross-Entropy-Loss von Grund auf. Bei kleinen Logits funktioniert es. Sobald Logits über 100 liegen, gibt es `inf`. Das Softmax ist übergelaufen, weil `exp(100)` größer ist, als float32 darstellen kann. Jedes ML-Framework löst das mit einem Zwei-Zeilen-Trick. Du kanntest den Trick nicht.

Numerische Stabilität ist kein theoretisches Thema. Sie entscheidet, ob ein Trainingslauf gelingt oder still scheitert. Jeder ernsthafte ML-Bug, den du irgendwann debuggen wirst, läuft auf Gleitkomma hinaus.

## Das Konzept

### IEEE 754: Wie Computer reelle Zahlen speichern

Computer speichern reelle Zahlen als Gleitkommawerte nach dem IEEE-754-Standard. Ein Float hat drei Teile: Vorzeichenbit, Exponent und Mantisse (Signifikand).

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

Die Mantisse bestimmt die Präzision (wie viele signifikante Stellen). Der Exponent bestimmt den Bereich (wie groß oder klein eine Zahl sein kann).

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 liefert etwa 7 Dezimalstellen Präzision. Es kann also 1.0000001 und 1.0000002 unterscheiden, aber nicht 1.00000001 und 1.00000002. Nach 7 Stellen ist alles Rundungsrauschen.

float16 liefert etwa 3 Stellen. Die größte darstellbare Zahl ist 65.504. Das ist in ML erschreckend klein, weil Logits, Gradienten und Aktivierungen das regelmäßig überschreiten.

bfloat16 ist Googles Antwort auf das Bereichsproblem von float16. Es hat denselben 8-Bit-Exponent wie float32 (gleicher Bereich bis 3.4e38), aber nur 7 Mantissen-Bits (weniger Präzision als float16). Für das Training neuronaler Netze ist der Bereich wichtiger als Präzision, deshalb gewinnt bfloat16 meist.

### Warum 0.1 + 0.2 != 0.3

Die Zahl 0.1 kann in binärem Gleitkomma nicht exakt dargestellt werden. In Basis 2 ist sie ein periodischer Bruch:

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

float32 kürzt das auf 23 Mantissen-Bits. Der gespeicherte Wert ist ungefähr 0.100000001490116. Genauso wird 0.2 als ungefähr 0.200000002980232 gespeichert. Die Summe ist 0.300000004470348, nicht 0.3.

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

Das ist für ML wichtig, weil:

1. Loss-Vergleiche wie `if loss < threshold` falsche Entscheidungen liefern können
2. Das Aufsummieren vieler kleiner Werte (Gradient-Updates über tausende Schritte) vom echten Wert wegdriftet
3. Checksummen und Reproduzierbarkeitstests scheitern, wenn Floats mit `==` verglichen werden

Die Lösung: Floats nie mit `==` vergleichen. Nutze `abs(a - b) < epsilon` oder `math.isclose()`.

### Katastrophale Auslöschung

Wenn du zwei fast gleiche Gleitkommazahlen subtrahierst, löschen sich die signifikanten Stellen aus, und Rundungsrauschen wird zu führenden Stellen.

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

Das ist 19 % relativer Fehler durch eine einzige Subtraktion. In ML passiert das immer dann, wenn du:

- Die Varianz von Daten mit großem Mittelwert berechnest: `E[x^2] - E[x]^2`, wenn E[x] groß ist
- Fast gleiche Log-Wahrscheinlichkeiten subtrahierst
- Finite-Differenzen-Gradienten mit zu kleinem Epsilon berechnest

Die Lösung: Formeln so umstellen, dass große, fast gleiche Zahlen nicht subtrahiert werden. Für Varianz: Welford-Algorithmus oder erst zentrieren. Für Log-Wahrscheinlichkeiten: durchgängig im Log-Raum rechnen.

### Overflow und Underflow

Overflow tritt auf, wenn ein Ergebnis zu groß für die Darstellung ist. Underflow tritt auf, wenn es zu klein ist (näher an null als die kleinste darstellbare positive Zahl).

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

Die `exp()`-Funktion ist die Hauptquelle für Overflow in ML:

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

Bei `log()` passiert es in die andere Richtung:

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

In ML taucht `exp()` in Softmax, Sigmoid und Wahrscheinlichkeitsberechnungen auf. `log()` taucht in Cross-Entropy, Log-Likelihoods und KL-Divergenz auf. Die Kombination `log(exp(x))` ist ohne die richtigen Tricks ein Minenfeld.

### Der Log-Sum-Exp-Trick

`log(sum(exp(x_i)))` direkt zu berechnen ist numerisch gefährlich. Wenn ein `x_i` groß ist, läuft `exp(x_i)` über. Wenn alle `x_i` stark negativ sind, underflowen alle `exp(x_i)` zu null und `log(0)` wird `-inf`.

Der Trick: Vor dem Exponentieren den Maximalwert abziehen.

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

Warum das funktioniert: Nach dem Abziehen von `max(x)` ist der größte Exponent `exp(0) = 1`. Overflow ist unmöglich. Mindestens ein Term in der Summe ist 1, also ist die Summe mindestens 1 und `log(1) = 0`. Underflow zu `-inf` ist unmöglich.

Beweis:

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

Setze `c = max(x)`, und Overflow ist eliminiert.

Dieser Trick steckt überall in ML:
- Softmax-Normalisierung
- Cross-Entropy-Loss-Berechnung
- Summation von Log-Wahrscheinlichkeiten in Sequenzmodellen
- Mixture of Gaussians
- Variational Inference

### Warum Softmax den Max-Subtraktions-Trick braucht

Softmax wandelt Logits in Wahrscheinlichkeiten um:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

Ohne Trick führen Logits [100, 101, 102] zu Overflow:

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

Mit Trick ziehst du max(x) = 102 ab:

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

Die Wahrscheinlichkeiten sind identisch. Die Rechnung ist sicher. Das ist keine Optimierung, sondern Voraussetzung für Korrektheit.

### NaN und Inf: Erkennen und Verhindern

`nan` (Not a Number) und `inf` (Unendlichkeit) verbreiten sich viral durch Berechnungen. Ein einzelnes `nan` in einem Gradient-Update macht das Gewicht zu `nan`, danach wird jeder Output `nan`. Das Training ist in einem Schritt kaputt.

So entsteht `inf`:
- `exp()` einer großen positiven Zahl
- Division durch null: `1.0 / 0.0`
- `float32`-Overflow in Akkumulationen

So entsteht `nan`:
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- `sqrt()` einer negativen Zahl
- `log()` einer negativen Zahl
- Jede Arithmetik mit vorhandenem `nan`

Erkennung:

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

Strategien zur Vermeidung:

1. Inputs für `exp()` clampen: `exp(clamp(x, -80, 80))`
2. Epsilon zu Nennern addieren: `x / (y + 1e-8)`
3. Epsilon in `log()` addieren: `log(x + 1e-8)`
4. Stabile Implementierungen nutzen (log-sum-exp, stabiles softmax)
5. Gradient Clipping gegen Gewichtsexplosion
6. Beim Debuggen nach jedem Forward-Pass auf `nan`/`inf` prüfen

### Numerische Gradientenprüfung

Analytische Gradienten (aus Backpropagation) können Fehler haben. Numerische Gradientenprüfung verifiziert sie über finite Differenzen.

Die zentrierte Differenzformel:

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

Sie hat Genauigkeit O(h^2), deutlich besser als die Vorwärtsdifferenz `(f(x+h) - f(x)) / h` mit nur O(h).

Wahl von h: Zu groß, dann ist die Approximation schlecht. Zu klein, dann zerstört katastrophale Auslöschung das Ergebnis. Typisch ist `h = 1e-5` bis `1e-7`.

Die Prüfung: Berechne den relativen Unterschied zwischen analytischen und numerischen Gradienten.

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

Faustregeln:
- relative_error < 1e-7: perfekt, Gradient korrekt
- relative_error < 1e-5: akzeptabel, vermutlich korrekt
- relative_error > 1e-3: etwas stimmt nicht
- relative_error > 1: Gradient ist komplett falsch

Prüfe Gradienten immer, wenn du eine neue Schicht oder Loss-Funktion implementierst. PyTorch bietet dafür `torch.autograd.gradcheck()`.

### Mixed-Precision-Training

Moderne GPUs haben Spezialhardware (Tensor Cores), die float16-Matrizenmultiplikation 2–8x schneller als float32 berechnet. Mixed Precision nutzt das:

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

Das Problem bei reinem float16-Training: Gradienten sind oft sehr klein (1e-8 oder kleiner). float16 underflowt alles unter ~6e-8 zu null. Das Modell lernt nicht mehr, weil alle Gradient-Updates null sind.

Die Lösung ist Loss Scaling:

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

Dynamisches Loss Scaling passt den Faktor automatisch an. Starte groß (65536). Wenn Gradienten zu `inf` überlaufen, halbiere. Wenn N Schritte ohne Overflow laufen, verdopple.

### bfloat16 vs float16: Warum bfloat16 fürs Training gewinnt

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 hat mehr Präzision (10 Mantissen-Bits statt 7), aber einen kleinen Bereich (max ~65.504). bfloat16 hat weniger Präzision, aber denselben Bereich wie float32 (max ~3.4e38).

Für das Training neuronaler Netze:

- Aktivierungen und Logits überschreiten bei Trainingsspitzen regelmäßig 65.504. float16 läuft über, bfloat16 nicht.
- Loss Scaling ist bei float16 nötig, bei bfloat16 wegen des größeren Bereichs meist nicht.
- bfloat16 ist eine einfache Kürzung von float32: die unteren 16 Mantissen-Bits wegwerfen. Die Umwandlung ist trivial und im Exponenten verlustfrei.

float16 ist oft besser für Inferenz, wo Werte begrenzt sind und Präzision wichtiger ist. bfloat16 ist fürs Training besser, wo der Bereich wichtiger ist. Deshalb unterstützen TPUs und moderne NVIDIA-GPUs (A100, H100) bfloat16 nativ.

### Gradient Clipping

Explodierende Gradienten entstehen, wenn Gradienten über viele Schichten exponentiell wachsen (häufig in RNNs, tiefen Netzen und Transformern). Ein einzelner großer Gradient kann in einem Schritt alle Gewichte beschädigen.

Zwei Arten von Clipping:

**Clip by value:** clamp jedes Gradient-Element unabhängig.

```
grad = clamp(grad, -max_val, max_val)
```

Einfach, kann aber die Richtung des Gradientenvektors verändern.

**Clip by norm:** skaliert den gesamten Gradientenvektor, sodass seine Norm einen Grenzwert nicht überschreitet.

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

Erhält die Richtung des Gradienten. Genau das macht `torch.nn.utils.clip_grad_norm_()`. Das ist der Standard.

Typische Werte: `max_norm=1.0` für Transformer, `max_norm=0.5` für RL, `max_norm=5.0` für einfachere Netze.

Gradient Clipping ist kein Hack. Es ist ein Sicherheitsmechanismus. Ohne ihn kann ein einzelner Ausreißer-Batch einen Gradienten erzeugen, der Wochen Training zerstört.

### Normalisierungsschichten als numerische Stabilisatoren

Batch Normalization, Layer Normalization und RMS Normalization werden meist als Regularisierer vorgestellt, die die Konvergenz verbessern. Sie sind auch numerische Stabilisatoren.

Ohne Normalisierung können Aktivierungen über Schichten exponentiell wachsen oder schrumpfen:

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

Normalisierung zentriert und skaliert Aktivierungen in jeder Schicht neu:

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

Das `epsilon` (typisch 1e-5) verhindert Division durch null, wenn alle Aktivierungen gleich sind. Die gelernten Parameter `gamma` und `beta` erlauben dem Netz, jede benötigte Skalierung wiederherzustellen.

So bleiben Werte im gesamten Netz in einem numerisch sicheren Bereich und verhindern sowohl Overflow im Forward-Pass als auch Gradientenexplosion im Backward-Pass.

### Häufige numerische ML-Bugs

**Bug: Loss ist nach einigen Epochen NaN.**
Ursache: Logits wurden zu groß, Softmax ist übergelaufen. Oder die Lernrate ist zu hoch und Gewichte divergieren.
Fix: stabiles Softmax (Max-Subtraktion), Lernrate senken, Gradient Clipping ergänzen.

**Bug: Loss bleibt bei log(num_classes).**
Ursache: Modell-Ausgaben sind nahezu gleichverteilte Wahrscheinlichkeiten. Oft sind Gradienten verschwunden oder das Modell lernt gar nicht.
Fix: Prüfe Labels, verifiziere die Loss-Funktion, prüfe auf tote ReLUs.

**Bug: Validierungsgenauigkeit liegt 1–3 % unter Erwartung.**
Ursache: Mixed Precision ohne korrektes Loss Scaling. Gradient-Underflow setzt kleine Updates still auf null.
Fix: Dynamisches Loss Scaling aktivieren oder auf bfloat16 wechseln.

**Bug: Gradient-Normen sind in manchen Schichten 0.0.**
Ursache: Tote ReLU-Neuronen (alle Inputs negativ) oder float16-Underflow.
Fix: LeakyReLU oder GELU nutzen, Gradient Scaling verwenden, Gewichtsinitialisierung prüfen.

**Bug: Modell läuft auf einer GPU, liefert auf anderer aber andere Ergebnisse.**
Ursache: Nichtdeterministische Reihenfolge bei Gleitkomma-Akkumulation. Parallele GPU-Reduktionen summieren in anderer Reihenfolge, und Gleitkomma-Addition ist nicht assoziativ.
Fix: Kleine Unterschiede (1e-6) akzeptieren oder `torch.use_deterministic_algorithms(True)` setzen und den Geschwindigkeitsverlust akzeptieren.

**Bug: `exp()` gibt in der Loss-Berechnung `inf` zurück.**
Ursache: Rohe Logits wurden ohne Max-Subtraktions-Trick an `exp()` übergeben.
Fix: `torch.nn.functional.log_softmax()` verwenden, das intern log-sum-exp nutzt.

**Bug: Training divergiert nach Umstieg von float32 auf float16.**
Ursache: float16 kann weder Gradienten kleiner als 6e-8 noch Aktivierungen größer als 65.504 darstellen.
Fix: Mixed Precision mit Loss Scaling (AMP) oder stattdessen bfloat16.

## Umsetzung

### Schritt 1: Grenzen der Gleitkomma-Präzision zeigen

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### Schritt 2: Naives vs. stabiles Softmax implementieren

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### Schritt 3: Stabiles Log-Sum-Exp implementieren

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### Schritt 4: Stabile Cross-Entropy implementieren

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### Schritt 5: Gradientenprüfung

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## In der Praxis

### Mixed-Precision-Simulation

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### Gradient Clipping

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf-Erkennung

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

Siehe `code/numerical.py` für vollständige Implementierungen mit allen demonstrierten Randfällen.

## Fertigstellen

Diese Lektion erzeugt:
- `code/numerical.py` mit stabilem Softmax, Log-Sum-Exp, Cross-Entropy, Gradientenprüfung und Mixed-Precision-Simulation
- `outputs/prompt-numerical-debugger.md` zum Diagnostizieren von NaN/Inf und numerischen Problemen im Training

Diese stabilen Implementierungen erscheinen in Phase 3 beim Bau der Trainingsschleife und in Phase 4 bei der Implementierung von Attention-Mechanismen wieder.

## Übungen

1. **Katastrophale Auslöschung.** Berechne die Varianz von [1000000.0, 1000001.0, 1000002.0] mit der naiven Formel `E[x^2] - E[x]^2` in float32. Berechne sie dann mit Welfords Online-Algorithmus. Vergleiche die Fehler mit der echten Varianz (0.6667).

2. **Präzisionssuche.** Finde den kleinsten positiven float32-Wert `x`, sodass `1.0 + x == 1.0` in Python gilt. Das ist das Machine Epsilon. Verifiziere, dass es zu `numpy.finfo(numpy.float32).eps` passt.

3. **Log-Sum-Exp-Randfälle.** Teste deine Funktion `logsumexp_stable` mit: (a) allen gleichen Werten, (b) einem Wert, der viel größer als der Rest ist, (c) allen stark negativen Werten (-1000). Verifiziere, dass sie korrekte Ergebnisse liefert, wo die naive Version scheitert.

4. **Gradientenprüfung einer neuronalen Netzwerkschicht.** Implementiere eine einzelne lineare Schicht `y = Wx + b` und ihren analytischen Backward-Pass. Nutze `numerical_gradient`, um die Korrektheit für eine 3x2-Gewichtsmatrix zu prüfen.

5. **Loss-Scaling-Experiment.** Simuliere Training mit float16: Erzeuge zufällige Gradienten im Bereich [1e-9, 1e-3], konvertiere nach float16 und miss den Anteil, der null wird. Wende dann Loss Scaling an (mit 1024 multiplizieren), konvertiere nach float16, skaliere zurück und miss den Null-Anteil erneut.

## Schlüsselbegriffe

| Begriff | Was man sagt | Was es wirklich bedeutet |
|------|----------------|----------------------|
| IEEE 754 | „Der Float-Standard“ | Internationaler Standard für binäre Gleitkommaformate, Rundungsregeln und Spezialwerte (inf, nan). Jede moderne CPU und GPU implementiert ihn. |
| Machine epsilon | „Die Präzisionsgrenze“ | Der kleinste Wert e, für den in einem Float-Format gilt: 1.0 + e != 1.0. Für float32 etwa 1.19e-7. |
| Katastrophale Auslöschung | „Präzisionsverlust durch Subtraktion“ | Beim Subtrahieren nahezu gleicher Gleitkommazahlen löschen sich signifikante Stellen aus, und Rundungsrauschen dominiert das Ergebnis. |
| Overflow | „Zahl zu groß“ | Ein Ergebnis überschreitet den maximal darstellbaren Wert und wird zu inf. exp(89) überläuft float32. |
| Underflow | „Zahl zu klein“ | Ein Ergebnis liegt näher an null als die kleinste darstellbare positive Zahl und wird zu 0.0. exp(-104) underflowt float32. |
| Log-Sum-Exp-Trick | „Erst das Maximum abziehen“ | Berechnung von log(sum(exp(x))) durch Herausziehen von exp(max(x)), um Overflow und Underflow zu vermeiden. Verwendet in Softmax, Cross-Entropy und Log-Wahrscheinlichkeitsrechnung. |
| Stabiles Softmax | „Softmax, das nicht explodiert“ | Vor dem Exponentieren max(logits) abziehen. Numerisch identisches Ergebnis, Overflow ausgeschlossen. |
| Gradientenprüfung | „Backprop verifizieren“ | Analytische Gradienten aus Backpropagation mit numerischen Gradienten aus finiten Differenzen vergleichen, um Implementierungsfehler zu finden. |
| Mixed Precision | „Float16 vorwärts, Float32 rückwärts“ | Niedrigpräzise Floats für schnelle Operationen und höherpräzise Floats für numerisch empfindliche Operationen verwenden. Typischer Speedup: 2–3x. |
| Loss Scaling | „Gradient-Underflow verhindern“ | Loss vor Backprop mit großer Konstante multiplizieren, damit Gradienten im darstellbaren float16-Bereich bleiben, dann vor dem Update wieder dividieren. |
| bfloat16 | „Brain Floating Point“ | Googles 16-Bit-Format mit 8 Exponent-Bits (gleicher Bereich wie float32) und 7 Mantissen-Bits (weniger Präzision als float16). Bevorzugt fürs Training. |
| Gradient Clipping | „Gradientennorm deckeln“ | Gradientenvektor so skalieren, dass seine Norm einen Schwellwert nicht überschreitet. Verhindert, dass explodierende Gradienten Gewichte zerstören. |
| NaN | „Not a Number“ | Spezieller Float-Wert aus undefinierten Operationen (0/0, inf-inf, sqrt(-1)). Propagiert durch alle folgenden Rechenoperationen. |
| Inf | „Unendlich“ | Spezieller Float-Wert durch Overflow oder Division durch null. Kann mit anderen Operationen NaN erzeugen (inf - inf, inf * 0). |
| Numerischer Gradient | „Brute-Force-Ableitung“ | Ableitung approximieren, indem f(x+h) und f(x-h) ausgewertet und durch 2h geteilt wird. Langsam, aber zuverlässig zur Verifikation. |

## Weiterführende Literatur

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- Die maßgebliche Referenz, dicht aber vollständig
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- Das NVIDIA-Paper, das Loss Scaling für float16-Training etabliert hat
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- Praktischer Leitfaden für Mixed Precision in PyTorch
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- Warum Google dieses Format für TPUs gewählt hat
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- Algorithmus zur Reduktion von Rundungsfehlern bei Gleitkomma-Summen
