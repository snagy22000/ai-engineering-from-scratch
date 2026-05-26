# Wahrscheinlichkeit und Verteilungen

> Wahrscheinlichkeit ist die Sprache, mit der KI Unsicherheit ausdrückt.

**Typ:** Lernen
**Sprachen:** Python
**Voraussetzungen:** Phase 1, Lektionen 01–04
**Zeit:** ~75 Minuten

## Lernziele

- PMFs und PDFs für Bernoulli-, kategoriale, Poisson-, Gleichverteilungs- und Normalverteilungen von Grund auf implementieren
- Erwartungswert, Varianz und den Zentralen Grenzwertsatz berechnen und erklären, warum Gauß-Verteilungen dominieren
- Softmax- und log-softmax-Funktionen mit dem Trick für numerische Stabilität bauen (maximalen Logit abziehen)
- Cross-Entropy-Loss aus Logits berechnen und mit der negativen Log-Likelihood verknüpfen

## Das Problem

Ein Klassifikator gibt `[0.03, 0.91, 0.06]` aus. Ein Sprachmodell wählt das nächste Wort aus 50.000 Kandidaten. Ein Diffusionsmodell erzeugt Bilder, indem es aus gelernten Verteilungen sampelt. All das ist Wahrscheinlichkeit in Aktion.

Jede Vorhersage eines Modells ist eine Wahrscheinlichkeitsverteilung. Jede Loss-Funktion misst, wie weit die vorhergesagte Verteilung von der wahren entfernt ist. Jeder Trainingsschritt passt Parameter so an, dass eine Verteilung einer anderen ähnlicher wird. Ohne Wahrscheinlichkeit kannst du kein einziges ML-Paper lesen, kein einziges Modell debuggen und nicht verstehen, warum dein Training-Loss `NaN` ist.

## Das Konzept

### Ereignisse, Stichprobenräume und Wahrscheinlichkeit

Der Stichprobenraum S ist die Menge aller möglichen Ergebnisse. Ein Ereignis ist eine Teilmenge des Stichprobenraums. Wahrscheinlichkeit ordnet Ereignissen Zahlen zwischen 0 und 1 zu.

```
Münzwurf:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Einzelner Würfelwurf:
  S = {1, 2, 3, 4, 5, 6}
  P(gerade) = P({2, 4, 6}) = 3/6 = 0.5
```

Drei Axiome definieren die gesamte Wahrscheinlichkeit:
1. P(A) >= 0 für jedes Ereignis A
2. P(S) = 1 (irgendetwas passiert immer)
3. P(A or B) = P(A) + P(B), wenn A und B nicht beide eintreten können

Alles andere (Satz von Bayes, Erwartungswerte, Verteilungen) folgt aus diesen drei Regeln.

### Bedingte Wahrscheinlichkeit und Unabhängigkeit

P(A|B) ist die Wahrscheinlichkeit von A unter der Bedingung, dass B eingetreten ist.

```
P(A|B) = P(A and B) / P(B)

Beispiel: Kartendeck
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

Zwei Ereignisse sind unabhängig, wenn das Wissen über das eine nichts über das andere verrät:

```
Unabhängig:   P(A|B) = P(A)
Äquivalent:   P(A and B) = P(A) * P(B)
```

Münzwürfe sind unabhängig. Karten ohne Zurücklegen zu ziehen nicht.

### Probability Mass Functions vs Probability Density Functions

Diskrete Zufallsvariablen haben eine Probability Mass Function (PMF). Jedes Ergebnis hat eine konkrete Wahrscheinlichkeit, die man direkt ablesen kann.

```
PMF: P(X = k)

Fairer Würfel:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Summe aller Wahrscheinlichkeiten = 1
```

Stetige Zufallsvariablen haben eine Probability Density Function (PDF). Die Dichte an einem einzelnen Punkt ist keine Wahrscheinlichkeit. Wahrscheinlichkeit entsteht, indem man die Dichte über ein Intervall integriert.

```
PDF: f(x)

P(a <= X <= b) = Integral von f(x) von a bis b

f(x) kann größer als 1 sein (Dichte, keine Wahrscheinlichkeit)
Integral von -inf bis +inf von f(x) dx = 1
```

Diese Unterscheidung ist in ML wichtig. Klassifikationsausgaben sind PMFs (diskrete Entscheidungen). Latente Räume in VAEs verwenden PDFs (stetig).

### Häufige Verteilungen

**Bernoulli:** ein Versuch, zwei Ergebnisse. Modelliert binäre Klassifikation.

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Kategorial:** ein Versuch, k Ergebnisse. Modelliert Mehrklassen-Klassifikation (Softmax-Ausgabe).

```
P(X = i) = p_i,  wobei sum of p_i = 1
Beispiel: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Gleichverteilung:** alle Ergebnisse sind gleich wahrscheinlich. Wird für zufällige Initialisierung verwendet.

```
Diskret: P(X = k) = 1/n für k in {1, ..., n}
Stetig: f(x) = 1/(b-a) für x in [a, b]
```

**Normalverteilung (Gaussian):** die Glockenkurve. Parametrisiert durch Mittelwert (mu) und Varianz (sigma^2).

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standardnormalverteilung: mu = 0, sigma = 1
  68% der Daten innerhalb von 1 sigma
  95% innerhalb von 2 sigma
  99.7% innerhalb von 3 sigma
```

**Poisson:** Anzahl seltener Ereignisse in einem festen Intervall. Modelliert Ereignisraten.

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### Erwartungswert und Varianz

Der Erwartungswert ist das gewichtete Durchschnittsergebnis.

```
Diskret:   E[X] = sum of x_i * P(X = x_i)
Stetig:    E[X] = Integral von x * f(x) dx
```

Die Varianz misst die Streuung um den Mittelwert.

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standardabweichung = sqrt(Var(X))
```

In ML erscheint der Erwartungswert als Loss-Funktion (durchschnittlicher Loss über die Datenverteilung). Varianz sagt dir etwas über die Stabilität des Modells. Hohe Varianz in Gradienten bedeutet rauschiges Training.

### Gemeinsame und marginale Verteilungen

Eine gemeinsame Verteilung P(X, Y) beschreibt zwei Zufallsvariablen zusammen.

Gemeinsames PMF-Beispiel (X = Wetter, Y = Regenschirm):

| | Y=0 (kein Regenschirm) | Y=1 (Regenschirm) | Marginale P(X) |
|---|---|---|---|
| X=0 (Sonne) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (Regen) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **Marginale P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

Die marginale Verteilung summiert die andere Variable heraus:

```
P(X = x) = Summe über alle y von P(X = x, Y = y)
```

Die Zeilen- und Spaltensummen in der Tabelle oben sind die Marginalen.

### Warum die Normalverteilung überall auftaucht

Der Zentrale Grenzwertsatz: Die Summe (oder der Mittelwert) vieler unabhängiger Zufallsvariablen konvergiert gegen eine Normalverteilung – unabhängig von der ursprünglichen Verteilung.

```
1 Würfelwurf:  Gleichverteilung (flach)
Mittelwert von 2 Würfeln:  dreieckig (spitz)
Mittelwert von 30 Würfeln: fast perfekte Glockenkurve

Das funktioniert für JEDE Ausgangsverteilung.
```

Deshalb gilt:
- Messfehler sind näherungsweise normalverteilt (viele kleine unabhängige Quellen)
- Gewichtsinitialisierungen in neuronalen Netzen verwenden Normalverteilungen
- Gradientenrauschen in SGD ist näherungsweise normalverteilt (Summe vieler Sample-Gradienten)
- Die Normalverteilung ist die Maximum-Entropy-Verteilung für gegebenen Mittelwert und gegebene Varianz

### Log-Wahrscheinlichkeiten

Rohwahrscheinlichkeiten verursachen numerische Probleme. Multipliziert man viele kleine Wahrscheinlichkeiten miteinander, läuft das Ergebnis schnell gegen null unter.

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (Unterlauf nach ~30 Termen)
```

Log-Wahrscheinlichkeiten beheben das. Multiplikationen werden zu Additionen.

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> endliche Zahl (kein Unterlauf)
```

Regeln:
- log(a * b) = log(a) + log(b)
- Log-Wahrscheinlichkeiten sind immer <= 0 (weil 0 < P <= 1)
- Negativer = unwahrscheinlicher
- Cross-Entropy-Loss ist die negative Log-Wahrscheinlichkeit der korrekten Klasse

### Softmax als Wahrscheinlichkeitsverteilung

Neuronale Netze geben rohe Scores (Logits) aus. Softmax verwandelt sie in eine gültige Wahrscheinlichkeitsverteilung.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Eigenschaften:
  - Alle Ausgaben liegen in (0, 1)
  - Alle Ausgaben summieren sich zu 1
  - Erhält die relative Reihenfolge der Eingaben
  - exp() verstärkt Unterschiede zwischen Logits
```

Der Softmax-Trick: Ziehe vor dem Exponentieren den größten Logit ab, um Overflow zu vermeiden.

```
z = [100, 101, 102]
exp(102) = Overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (sicher)

Gleiches Ergebnis, kein Overflow.
```

Log-softmax kombiniert Softmax und Logarithmus für numerische Stabilität. PyTorch verwendet das intern für Cross-Entropy-Loss.

### Sampling

Sampling bedeutet, zufällige Werte aus einer Verteilung zu ziehen. In ML:
- Dropout sampelt zufällig, welche Neuronen auf null gesetzt werden
- Data Augmentation sampelt zufällige Transformationen
- Sprachmodelle samplen das nächste Token aus der vorhergesagten Verteilung
- Diffusionsmodelle samplen Rauschen und entrauschen es schrittweise

Sampling aus beliebigen Verteilungen erfordert Techniken wie Inverse-Transform-Sampling, Rejection Sampling oder den Reparameterization Trick (verwendet in VAEs).

## Umsetzung

### Schritt 1: Wahrscheinlichkeitsgrundlagen

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### Schritt 2: PMF und PDF von Grund auf

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### Schritt 3: Erwartungswert und Varianz

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### Schritt 4: Aus Verteilungen sampeln

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### Schritt 5: Softmax und Log-Wahrscheinlichkeiten

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### Schritt 6: Demonstration des Zentralen Grenzwertsatzes

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### Schritt 7: Visualisierung

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

Vollständige Implementierungen mit allen Visualisierungen findest du in `code/probability.py`.

## In der Praxis

Mit NumPy und SciPy ist alles oben ein Einzeiler:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

Du hast das alles von Grund auf gebaut. Jetzt weißt du, was die Bibliotheksaufrufe wirklich tun.

## Übungen

1. Implementiere Inverse-Transform-Sampling für die Exponentialverteilung. Verifiziere es, indem du 10.000 Werte sampelst und das Histogramm mit der echten PDF vergleichst.

2. Baue eine gemeinsame Verteilungstabelle für zwei gezinkte Würfel. Berechne die marginalen Verteilungen und prüfe, ob die Würfel unabhängig sind.

3. Berechne den Cross-Entropy-Loss für einen 5-Klassen-Klassifikator, der Logits `[2.0, 0.5, -1.0, 3.0, 0.1]` ausgibt, wenn die korrekte Klasse Index 3 ist. Verifiziere dein Ergebnis dann mit PyTorchs `nn.CrossEntropyLoss`.

4. Schreibe eine Funktion, die eine Liste von Log-Wahrscheinlichkeiten nimmt und die wahrscheinlichste Sequenz, die gesamte Log-Wahrscheinlichkeit und die entsprechende rohe Wahrscheinlichkeit zurückgibt. Teste sie mit einem Satz aus 50 Wörtern, bei dem jedes Wort die Wahrscheinlichkeit 0.01 hat.

## Schlüsselbegriffe

| Begriff | Was man sagt | Was es wirklich bedeutet |
|------|----------------|----------------------|
| Stichprobenraum | „Alle Möglichkeiten“ | Die Menge S aller möglichen Ergebnisse eines Experiments |
| PMF | „Die Wahrscheinlichkeitsfunktion“ | Eine Funktion, die die exakte Wahrscheinlichkeit jedes diskreten Ergebnisses angibt und sich zu 1 aufsummiert |
| PDF | „Die Wahrscheinlichkeitskurve“ | Eine Dichtefunktion für stetige Variablen. Integriere sie über ein Intervall, um eine Wahrscheinlichkeit zu erhalten |
| Bedingte Wahrscheinlichkeit | „Wahrscheinlichkeit unter einer Bedingung“ | P(A\|B) = P(A and B) / P(B). Die Grundlage des bayesianischen Denkens und des Satzes von Bayes |
| Unabhängigkeit | „Sie beeinflussen sich nicht“ | P(A and B) = P(A) * P(B). Das Wissen über ein Ereignis verrät dir nichts über das andere |
| Erwartungswert | „Der Durchschnitt“ | Die wahrscheinlichkeitsgewichtete Summe aller Ergebnisse. Die Loss-Funktion ist ein Erwartungswert |
| Varianz | „Wie stark gestreut“ | Die erwartete quadrierte Abweichung vom Mittelwert. Hohe Varianz = rauschige, instabile Schätzungen |
| Normalverteilung | „Die Glockenkurve“ | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2)). Taucht wegen des ZGS überall auf |
| Zentraler Grenzwertsatz | „Mittelwerte werden normal“ | Der Mittelwert vieler unabhängiger Samples konvergiert gegen eine Normalverteilung – unabhängig von der Quelle |
| Gemeinsame Verteilung | „Zwei Variablen zusammen“ | P(X, Y) beschreibt die Wahrscheinlichkeit jeder Kombination von X- und Y-Ergebnissen |
| Marginale Verteilung | „Die andere Variable heraus summieren“ | P(X) = sum_y P(X, Y). Rekonstruiert die Verteilung einer Variable aus der gemeinsamen Verteilung |
| Log-Wahrscheinlichkeit | „Logarithmus der Wahrscheinlichkeit“ | log P(x). Macht Produkte zu Summen und verhindert numerischen Unterlauf in langen Sequenzen |
| Softmax | „Scores in Wahrscheinlichkeiten verwandeln“ | softmax(z_i) = exp(z_i) / sum(exp(z_j)). Bildet reellwertige Logits auf eine gültige Wahrscheinlichkeitsverteilung ab |
| Cross-Entropy | „Die Loss-Funktion“ | -sum(p_true * log(p_predicted)). Misst, wie unterschiedlich zwei Verteilungen sind. Kleiner ist besser |
| Logits | „Rohe Modellausgaben“ | Nicht normalisierte Scores vor Softmax. Der Name kommt von der logistischen Funktion |
| Sampling | „Zufallswerte ziehen“ | Werte gemäß einer Wahrscheinlichkeitsverteilung erzeugen. So erzeugen Modelle Ausgaben |

## Weiterführende Literatur

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) – visueller Beweis, warum Mittelwerte normal werden
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) – kompakte Referenz, die all das hier und mehr abdeckt
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) – warum numerische Stabilität wichtig ist und wie man sie erreicht
