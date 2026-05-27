# Sampling-Methoden

> Sampling ist, wie KI den Raum der Möglichkeiten erkundet.

**Typ:** Build
**Sprache:** Python
**Voraussetzungen:** Phase 1, Lektionen 06–07 (Wahrscheinlichkeit, Bayes-Theorem)
**Dauer:** ~120 Minuten

## Lernziele

- Inverse-CDF-, Rejection- und Importance-Sampling von Grund auf implementieren – nur mit gleichverteilten Zufallszahlen
- Temperature-, Top-k- und Top-p-(Nucleus-)Sampling für die Tokengenerierung in Sprachmodellen bauen
- Den Reparameterization Trick erklären und verstehen, warum er Backpropagation durch Sampling in VAEs ermöglicht
- Metropolis-Hastings-MCMC ausführen, um aus einer unnormalisierten Zielverteilung zu sampeln

## Das Problem

Ein Sprachmodell verarbeitet deinen Prompt und erzeugt einen Vektor mit 50.000 Logits. Einer pro Token im Vokabular. Jetzt muss es ein Token auswählen. Wie?

Wenn es immer das Token mit der höchsten Wahrscheinlichkeit nimmt, ist jede Antwort identisch. Deterministisch. Langweilig. Wenn es gleichverteilt zufällig auswählt, ist die Ausgabe Kauderwelsch. Die Lösung liegt zwischen diesen Extremen – und genau das steuert Sampling.

Sampling ist nicht nur für Textgenerierung relevant. Reinforcement Learning schätzt Policy-Gradienten durch das Sampeln von Trajektorien. VAEs lernen latente Repräsentationen, indem sie aus gelernten Verteilungen sampeln und durch den Zufall backpropagieren. Diffusionsmodelle erzeugen Bilder, indem sie Rauschen sampeln und iterativ entrauschen. Monte-Carlo-Methoden schätzen Integrale ohne geschlossene Lösung. MCMC-Algorithmen erkunden hochdimensionale Posteriorverteilungen, die man nicht vollständig aufzählen kann.

Jedes generative KI-System ist ein Sampling-System. Die gewählte Sampling-Strategie bestimmt Qualität, Vielfalt und Steuerbarkeit der Ausgabe. In dieser Lektion baust du die wichtigsten Sampling-Methoden von Grund auf – von gleichverteilten Zufallszahlen bis zu Techniken moderner LLMs und generativer Modelle.

## Das Konzept

### Warum Sampling wichtig ist

Sampling hat in KI und maschinellem Lernen vier zentrale Rollen:

**Generierung.** Sprachmodelle, Diffusionsmodelle und GANs erzeugen Ausgaben per Sampling. Der Algorithmus steuert direkt Kreativität, Kohärenz und Vielfalt. Temperature-, Top-k- und Nucleus-Sampling sind tägliche Stellschrauben in der Praxis.

**Training.** Stochastic Gradient Descent sampelt Mini-Batches. Dropout sampelt zu deaktivierende Neuronen. Data Augmentation sampelt zufällige Transformationen. Importance Sampling gewichtet Samples neu, um die Gradientenvarianz im Reinforcement Learning zu senken (PPO, TRPO).

**Schätzung.** Viele Größen im ML haben keine geschlossene Form: der erwartete Verlust über eine Datenverteilung, die Partition Function energie-basierter Modelle oder die Evidence in bayesianischer Inferenz. Monte-Carlo-Schätzung approximiert all das durch Mittelung über Samples.

**Exploration.** MCMC-Algorithmen erkunden Posteriorverteilungen in der bayesianischen Inferenz. Evolutionary Strategies sampeln Parameterstörungen. Thompson Sampling balanciert Exploration und Exploitation bei Bandits.

Die Kernherausforderung: Direkt sampeln kannst du nur aus einfachen Verteilungen (uniform, normalverteilt). Für alles andere brauchst du Methoden, die einfache Samples in Samples der Zielverteilung transformieren.

### Gleichverteiltes Zufallssampling

Hier startet jede Sampling-Methode. Ein Uniform-Zufallszahlengenerator erzeugt Werte in [0, 1), wobei jedes gleich lange Teilintervall die gleiche Wahrscheinlichkeit hat.

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

Um gleichverteilt aus einer diskreten Menge mit n Elementen zu sampeln, erzeugst du U und gibst floor(n * U) zurück. Für einen kontinuierlichen Bereich [a, b] berechnest du a + (b - a) * U.

Die zentrale Einsicht: Eine einzige gleichverteilte Zufallszahl enthält genau genug Zufall, um ein Sample aus beliebigen Verteilungen zu erzeugen. Der Trick ist die richtige Transformation.

### Inverse-CDF-Methode (Inverse Transform Sampling)

Die kumulative Verteilungsfunktion (CDF) ordnet Werten Wahrscheinlichkeiten zu:

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

Die inverse CDF ordnet Wahrscheinlichkeiten wieder Werten zu. Wenn U ~ Uniform(0, 1), dann folgt X = F_inverse(U) der Zielverteilung.

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**Beispiel Exponentialverteilung:**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

Das funktioniert perfekt, wenn sich F_inverse in geschlossener Form angeben lässt. Für die Normalverteilung gibt es keine geschlossene inverse CDF, daher nutzt man andere Methoden (Box-Muller oder numerische Approximation).

**Diskrete Variante:** Für diskrete Verteilungen baust du die CDF als kumulative Summe auf, erzeugst U und findest den ersten Index, bei dem die kumulative Summe U überschreitet. So funktioniert `sample_categorical` in Lektion 06.

### Rejection Sampling

Wenn du die CDF nicht invertieren kannst, aber die Ziel-PDF bis auf eine Konstante auswerten kannst, funktioniert Rejection Sampling.

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

Je enger die Schranke M, desto höher die Akzeptanzrate. In niedrigen Dimensionen (1–3) funktioniert Rejection Sampling gut. In hohen Dimensionen sinkt die Akzeptanzrate exponentiell, weil der Großteil des Proposal-Volumens verworfen wird. Das ist der Fluch der Dimensionalität beim Rejection Sampling.

**Beispiel: Sampling aus einer abgeschnittenen Normalverteilung.** Nutze ein gleichverteiltes Proposal über dem abgeschnittenen Bereich. Die Hülle M ist dort das Maximum der Normal-PDF.

**Beispiel: Sampling aus einem Halbkreis.** Schlage Punkte gleichverteilt im umschließenden Rechteck vor. Akzeptiere, wenn der Punkt im Halbkreis liegt. So berechnet Monte Carlo π: Die Akzeptanzrate entspricht dem Flächenverhältnis π/4.

### Importance Sampling

Manchmal brauchst du keine Samples aus der Zielverteilung p(x). Du willst einen Erwartungswert unter p(x) schätzen, hast aber Samples aus einer anderen Verteilung q(x).

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

Das ist entscheidend im Reinforcement Learning. In PPO (Proximal Policy Optimization) sammelst du Trajektorien unter einer alten Policy pi_old, willst aber eine neue Policy pi_new optimieren. Das Importance-Gewicht ist pi_new(a|s) / pi_old(a|s). PPO clippt diese Gewichte, damit die neue Policy nicht zu weit von der alten abweicht.

Die Varianz des Importance-Sampling-Schätzers hängt davon ab, wie ähnlich q und p sind. Wenn q stark von p abweicht, bekommen wenige Samples sehr große Gewichte und dominieren die Schätzung. Self-normalized Importance Sampling teilt durch die Summe der Gewichte, um dieses Problem zu reduzieren:

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### Monte-Carlo-Schätzung

Monte-Carlo-Schätzung approximiert Integrale durch das Mitteln zufälliger Samples. Das Gesetz der großen Zahlen garantiert Konvergenz.

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

Die Fehlerrate ist unabhängig von der Dimension. Deshalb dominieren Monte-Carlo-Methoden in hohen Dimensionen, in denen gitterbasierte Integration nicht praktikabel ist.

**π schätzen:**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**Erwartungswerte schätzen:**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### Markov Chain Monte Carlo (MCMC): Metropolis-Hastings

MCMC konstruiert eine Markovkette, deren stationäre Verteilung die Zielverteilung p(x) ist. Nach genügend Schritten sind Samples aus der Kette (näherungsweise) Samples aus p(x).

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

Für symmetrische Proposals (q(x'|x) = q(x|x')) vereinfacht sich das Verhältnis zu p(x')/p(x). Das ist der ursprüngliche Metropolis-Algorithmus.

**Warum es funktioniert.** Die Akzeptanzregel stellt Detailed Balance sicher: Die Wahrscheinlichkeit, bei x zu sein und nach x' zu gehen, ist gleich der Wahrscheinlichkeit, bei x' zu sein und nach x zu gehen. Detailed Balance impliziert, dass p(x) die stationäre Verteilung der Kette ist.

**Praktische Hinweise:**
- Burn-in: frühe Samples verwerfen, bevor die Kette ihr Gleichgewicht erreicht
- Thinning: nur jedes k-te Sample behalten, um Autokorrelation zu reduzieren
- Proposal-Skala: zu klein -> Kette bewegt sich langsam (hohe Akzeptanz, langsame Exploration); zu groß -> viele Ablehnungen (niedrige Akzeptanz, Kette bleibt hängen)
- Die optimale Akzeptanzrate für ein gaußsches Proposal in hohen Dimensionen liegt ungefähr bei 0,234

### Gibbs Sampling

Gibbs Sampling ist ein Spezialfall von MCMC für multivariate Verteilungen. Statt in allen Dimensionen gleichzeitig eine Bewegung vorzuschlagen, aktualisiert es jeweils nur eine Variable aus ihrer bedingten Verteilung.

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs Sampling setzt voraus, dass du aus jeder bedingten Verteilung p(x_i | x_{-i}) samplen kannst. Das ist bei vielen Modellen direkt möglich:
- Bayesian Networks: Bedingte Verteilungen folgen aus der Graphstruktur
- Gaussian Mixtures: Bedingte Verteilungen sind gaußsch
- Ising-Modelle: Die bedingte Verteilung jedes Spins hängt nur von seinen Nachbarn ab

Die Akzeptanzrate ist immer 1 (jeder Vorschlag wird akzeptiert), weil das Sampeln aus der exakten bedingten Verteilung Detailed Balance automatisch erfüllt.

**Einschränkung.** Bei stark korrelierten Variablen mischt Gibbs Sampling langsam, weil Ein-Variable-Updates keine großen diagonalen Sprünge durch die Verteilung machen können.

### Temperature Sampling (in LLMs verwendet)

Sprachmodelle geben Logits z_1, ..., z_V für jedes Token im Vokabular aus. Softmax wandelt sie in Wahrscheinlichkeiten um. Temperature skaliert die Logits vor Softmax neu:

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**Warum es funktioniert.** Das Teilen der Logits durch T < 1 verstärkt Unterschiede zwischen Logits. Wenn z_1 = 2 und z_2 = 1 ist, ergibt T = 0,5 die Werte z_1/T = 4 und z_2/T = 2; die Lücke wird größer. Nach Softmax bekommt das Token mit dem höchsten Logit einen deutlich größeren Anteil.

**In der Praxis:**
- T = 0.0: greedy decoding, gut für faktische Q&A
- T = 0.3–0.7: leicht kreativ, gut für Codegenerierung
- T = 0.7–1.0: ausgewogen, gut für allgemeine Unterhaltung
- T = 1.0–1.5: kreatives Schreiben, Brainstorming
- T > 1.5: zunehmend zufällig, selten nützlich

Temperature ändert nicht, welche Tokens möglich sind. Sie verändert die Wahrscheinlichkeitsmasse pro Token.

### Top-k Sampling

Top-k Sampling beschränkt die Kandidatenmenge auf die k Tokens mit den höchsten Wahrscheinlichkeiten, normalisiert dann neu und sampelt aus dieser eingeschränkten Menge.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k verhindert, dass das Modell extrem unwahrscheinliche Tokens auswählt (Tippfehler, Unsinn), die im langen Verteilungsschwanz liegen. Problem: k ist unabhängig vom Kontext fix. Wenn das Modell sehr sicher ist (ein Token hat 95 %), erlaubt k = 40 trotzdem 39 Alternativen. Wenn es unsicher ist (Wahrscheinlichkeit verteilt sich über 1000 Tokens), schneidet k = 40 plausible Optionen ab.

### Top-p (Nucleus) Sampling

Top-p Sampling passt die Größe der Kandidatenmenge dynamisch an. Statt eine feste Anzahl Tokens zu behalten, nimmt es die kleinste Menge, deren kumulierte Wahrscheinlichkeit p überschreitet.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

Wenn das Modell sicher ist, behält Nucleus Sampling wenige Tokens (vielleicht 2–3). Wenn das Modell unsicher ist, behält es viele (vielleicht 200). Dieses adaptive Verhalten ist der Grund, warum Nucleus Sampling häufig besseren Text liefert als Top-k.

**Häufige Kombinationen:**
- Temperature 0.7 + top-p 0.9: gute Allround-Einstellung
- Temperature 0.0 (greedy): am besten für deterministische Aufgaben
- Temperature 1.0 + top-k 50: Einstellung aus Fan et al. (2018)

Top-k und Top-p lassen sich kombinieren. Zuerst Top-k anwenden, dann Top-p auf die verbleibende Menge.

### Reparameterization Trick (in VAEs verwendet)

Variational Autoencoders (VAEs) lernen, indem sie Eingaben auf eine Verteilung im latenten Raum abbilden, daraus sampeln und das Sample zurückdekodieren. Das Problem: Durch eine Sampling-Operation kann man nicht direkt backpropagieren.

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

Der Reparameterization Trick trennt Zufall und Parameter:

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

Das funktioniert, weil N(mu, sigma^2) dieselbe Verteilung hat wie mu + sigma * N(0, 1). Die Kerneinsicht: Verlagere den Zufall in eine parameterfreie Quelle (epsilon) und drücke das Sample als differenzierbare Transformation der Parameter aus.

**Im VAE-Trainingsloop:**
1. Der Encoder gibt für jede Eingabe mu und log(sigma^2) aus
2. Sample epsilon ~ N(0, 1)
3. Berechne z = mu + sigma * epsilon
4. Dekodiere z zur Rekonstruktion der Eingabe
5. Backpropagiere durch die Schritte 4, 3, 2, 1 (möglich, weil Schritt 3 differenzierbar ist)

Ohne den Reparameterization Trick lassen sich VAEs nicht mit Standard-Backpropagation trainieren. Diese einzelne Idee machte VAEs praktikabel.

### Gumbel-Softmax (differenzierbares kategoriales Sampling)

Der Reparameterization Trick funktioniert für kontinuierliche Verteilungen (gaußsch). Für diskrete kategoriale Verteilungen braucht man einen anderen Ansatz. Gumbel-Softmax liefert eine differenzierbare Approximation für kategoriales Sampling.

**Der Gumbel-Max-Trick (nicht differenzierbar):**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax (differenzierbare Approximation):**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax erzeugt eine kontinuierliche Relaxation eines diskreten Samples. Die Ausgabe ist ein Wahrscheinlichkeitsvektor (softes One-Hot) statt eines harten One-Hot-Vektors. Gradienten fließen durch Softmax. Im Forward-Pass des Trainings kannst du den „straight-through“-Estimator nutzen: im Forward den harten Argmax, im Backward die weichen Gumbel-Softmax-Gradienten.

**Anwendungen:**
- Diskrete latente Variablen in VAEs
- Neural Architecture Search (Auswahl diskreter Operationen)
- Harte Attention-Mechanismen
- Reinforcement Learning mit diskreten Aktionen

### Stratified Sampling

Standard-Monte-Carlo-Sampling kann zufällig Lücken im Sample-Raum lassen. Stratified Sampling erzwingt gleichmäßige Abdeckung, indem der Raum in Strata geteilt wird und aus jedem Stratum gesampelt wird.

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

Stratified Sampling hat immer geringere oder gleiche Varianz wie Standard-Monte-Carlo:

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**Anwendungen:**
- Numerische Integration (Quasi-Monte-Carlo)
- Aufteilung von Trainingsdaten (Klassenbalance in jedem Fold)
- Importance Sampling mit Stratifikation (Kombination beider Techniken)
- NeRF (Neural Radiance Fields) nutzt Stratified Sampling entlang von Kamerastrahlen

### Verbindung zu Diffusionsmodellen

Diffusionsmodelle erzeugen Bilder über einen Sampling-Prozess. Der Vorwärtsprozess fügt einem Bild über T Schritte gaußsches Rauschen hinzu, bis nur noch reines Rauschen bleibt. Der Rückwärtsprozess lernt das Entrauschen und rekonstruiert das Bild Schritt für Schritt.

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

Die Verbindung zu den Methoden dieser Lektion:
- Jeder Entrauschungsschritt nutzt den Reparameterization Trick (Rauschen sampeln, deterministische Transformation anwenden)
- Der Noise Schedule {alpha_t} steuert eine Form von Temperature Annealing
- Das Training nutzt Monte-Carlo-Schätzung zur Approximation der ELBO (evidence lower bound)
- Ancestral Sampling in Diffusionsmodellen ist eine Markovkette (jeder Schritt hängt nur vom aktuellen Zustand ab)

Der gesamte Bildgenerierungsprozess ist iteratives Sampling: Starte mit Rauschen und sample in jedem Schritt eine leicht weniger verrauschte Version, bedingt auf dem gelernten Denoising-Modell.

## Baue es

### Schritt 1: Uniform- und inverse-CDF-Sampling

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

Erzeuge 10.000 Exponential-Samples und verifiziere, dass der Mittelwert 1/lambda ist.

### Schritt 2: Rejection Sampling

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

Nutze Rejection Sampling, um aus einer abgeschnittenen Normalverteilung zu ziehen. Verifiziere die Form mit einem Histogramm der Samples.

### Schritt 3: Importance Sampling

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

Schätze E[X^2] unter einer Normalverteilung mit einem Uniform-Proposal. Vergleiche mit der bekannten Lösung (mu^2 + sigma^2).

### Schritt 4: Monte-Carlo-Schätzung von pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### Schritt 5: Metropolis-Hastings-MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

Sample aus einer bimodalen Verteilung (Mischung aus zwei Gaußverteilungen). Visualisiere die Trajektorie der Kette.

### Schritt 6: Gibbs Sampling

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### Schritt 7: Temperature Sampling

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

Zeige, wie Temperature die Ausgabeverteilung für eine Menge Token-Logits verändert.

### Schritt 8: Top-k- und Top-p-Sampling

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### Schritt 9: Reparameterization Trick

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

Zeige, dass Gradienten durch das reparameterisierte Sample fließen, aber nicht durch direktes Sampling.

### Schritt 10: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

Zeige, wie eine sinkende Temperature die Ausgabe an einen One-Hot-Vektor annähert.

Vollständige Implementierungen inklusive aller Visualisierungen findest du in `code/sampling.py`.

## Verwenden

Mit NumPy und SciPy lauten die produktionsnahen Varianten:

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

Für MCMC im größeren Maßstab nutze spezialisierte Bibliotheken:
- PyMC: vollständiges bayesianisches Modellieren mit NUTS (adaptives HMC)
- emcee: Ensemble-MCMC-Sampler
- NumPyro/JAX: GPU-beschleunigtes MCMC

Du hast diese Methoden von Grund auf gebaut. Jetzt weißt du, was die Library-Calls wirklich machen.

## Übungen

1. Implementiere Inverse-CDF-Sampling für die Cauchy-Verteilung. Die CDF ist F(x) = 0.5 + arctan(x)/pi. Erzeuge 10.000 Samples und plotte das Histogramm gegen die echte PDF. Achte auf die schweren Tails (extreme Werte weit weg vom Zentrum).

2. Nutze Rejection Sampling, um Samples aus einer Beta(2, 5)-Verteilung mit einem Uniform(0, 1)-Proposal zu erzeugen. Plotte die akzeptierten Samples gegen die echte Beta-PDF. Wie hoch ist die theoretische Akzeptanzrate?

3. Schätze das Integral von sin(x) von 0 bis pi per Monte Carlo mit 1.000, 10.000 und 100.000 Samples. Vergleiche den Fehler auf jedem Niveau. Verifiziere, dass der Fehler wie O(1/sqrt(N)) skaliert.

4. Implementiere Metropolis-Hastings, um aus einer 2D-Verteilung p(x, y) proportional zu exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2) zu samplen. Plotte die Samples und die Trajektorie der Kette. Experimentiere mit verschiedenen Proposal-Standardabweichungen.

5. Baue eine vollständige Textgenerierungs-Demo: Gegeben ein Vokabular aus 10 Wörtern mit Logits, generiere Sequenzen aus 20 Tokens mit (a) greedy, (b) temperature=0.7, (c) top-k=3, (d) top-p=0.9. Vergleiche die Vielfalt der Ausgaben über 5 Läufe.

## Schlüsselbegriffe

| Begriff | Was man oft sagt | Was es tatsächlich bedeutet |
|------|----------------|----------------------|
| Sampling | „Zufallswerte ziehen“ | Werte gemäß einer Wahrscheinlichkeitsverteilung erzeugen. Der Mechanismus hinter generativer KI |
| Uniformverteilung | „Alle gleich wahrscheinlich“ | Jeder Wert in [a, b] hat die gleiche Wahrscheinlichkeitsdichte 1/(b-a). Der Ausgangspunkt aller Sampling-Methoden |
| Inverse CDF | „Wahrscheinlichkeitstransformation“ | F_inverse(U) wandelt ein Uniform-Sample in ein Sample aus jeder Verteilung mit bekannter CDF um. Exakt und effizient |
| Rejection Sampling | „Vorschlagen und akzeptieren/ablehnen“ | Aus einem einfachen Proposal erzeugen, mit Wahrscheinlichkeit proportional zum Verhältnis Ziel/Proposal akzeptieren. Exakt, aber verschwenderisch |
| Importance Sampling | „Samples neu gewichten“ | Erwartungswerte unter p(x) mit Samples aus q(x) schätzen, indem jedes Sample mit p(x)/q(x) gewichtet wird. Zentral für PPO in RL |
| Monte Carlo | „Zufallssamples mitteln“ | Integrale als Mittelwerte von Samples approximieren. Fehler O(1/sqrt(N)) unabhängig von der Dimension |
| MCMC | „Random Walk mit Konvergenz“ | Eine Markovkette konstruieren, deren stationäre Verteilung die Zielverteilung ist. Metropolis-Hastings ist der grundlegende Algorithmus |
| Metropolis-Hastings | „Bergauf akzeptieren, bergab manchmal“ | Bewegungen vorschlagen und anhand des Dichteverhältnisses akzeptieren. Detailed Balance sichert Konvergenz zur Zielverteilung |
| Gibbs Sampling | „Eine Variable nach der anderen“ | Jede Variable aus ihrer bedingten Verteilung aktualisieren, während die anderen fix bleiben. 100 % Akzeptanzrate |
| Temperature | „Konfidenz-Regler“ | Teilt Logits vor Softmax durch T. T<1 schärft (mehr Konfidenz), T>1 glättet (mehr Vielfalt) |
| Top-k Sampling | „Die k besten behalten“ | Alle außer den k wahrscheinlichsten Tokens auf null setzen, neu normalisieren, sampeln. Feste Kandidatenmenge |
| Nucleus Sampling (top-p) | „Die wahrscheinlichen behalten“ | Kleinste Tokenmenge behalten, deren kumulierte Wahrscheinlichkeit p übersteigt. Adaptive Kandidatenmenge |
| Reparameterization Trick | „Zufall nach außen verlagern“ | Schreibe z = mu + sigma * epsilon mit epsilon ~ N(0,1). Macht Sampling differenzierbar. Essenziell fürs VAE-Training |
| Gumbel-Softmax | „Softes kategoriales Sampling“ | Differenzierbare Approximation für kategoriales Sampling mit Gumbel-Rauschen + Softmax mit Temperature |
| Stratified Sampling | „Erzwungene Abdeckung“ | Sample-Raum in Strata teilen, aus jedem Stratum sampeln. Immer geringere Varianz als naives Monte Carlo |
| Burn-in | „Aufwärmphase“ | Frühe MCMC-Samples, die verworfen werden, bevor die Kette ihre stationäre Verteilung erreicht |
| Detailed Balance | „Reversibilitätsbedingung“ | p(x) * T(x->y) = p(y) * T(y->x). Hinreichende Bedingung dafür, dass p stationäre Verteilung einer Markovkette ist |
| Diffusion Sampling | „Iteratives Entrauschen“ | Daten erzeugen, indem man mit Rauschen startet und gelernte Entrauschungsschritte anwendet. Jeder Schritt ist eine bedingte Sampling-Operation |

## Weiterführende Literatur

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - detailliertes Tutorial zu MCMC-Grundlagen
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - Originalarbeit zu Gumbel-Softmax
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - Arbeit zu Nucleus-(Top-p-)Sampling
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - VAE-Paper mit Einführung des Reparameterization Tricks
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM verbindet Sampling mit Bildgenerierung
