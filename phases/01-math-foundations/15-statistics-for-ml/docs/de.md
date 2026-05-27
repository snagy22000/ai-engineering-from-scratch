# Statistik für Machine Learning

> Statistik zeigt dir, ob dein Modell wirklich funktioniert oder nur Glück hatte.

**Typ:** Build
**Sprache:** Python
**Voraussetzungen:** Phase 1, Lektionen 06 (Wahrscheinlichkeit und Verteilungen), 07 (Satz von Bayes)
**Zeit:** ~120 Minuten

## Lernziele

- Deskriptive Statistiken, Pearson-/Spearman-Korrelation und Kovarianzmatrizen von Grund auf berechnen
- Hypothesentests (t-Test, Chi-Quadrat) durchführen und p-Werte sowie Konfidenzintervalle korrekt interpretieren
- Mit Bootstrap-Resampling Konfidenzintervalle für beliebige Metriken ohne Verteilungsannahmen erstellen
- Statistische Signifikanz mit Effektstärkemaßen von praktischer Signifikanz unterscheiden

## Das Problem

Du hast zwei Modelle trainiert. Modell A erreicht 0,87 auf deinem Testset. Modell B erreicht 0,89. Du deployst Modell B. Drei Wochen später sind die Produktionsmetriken schlechter als vorher. Was ist passiert?

Modell B war Modell A in Wirklichkeit nicht überlegen. Die Differenz von 0,02 war Rauschen. Dein Testset war zu klein, die Varianz zu hoch oder beides. Du hast Zufall als Verbesserung ausgeliefert.

Das passiert ständig: verschobene Kaggle-Leaderboards, Papers ohne Reproduzierbarkeit, A/B-Tests mit „Gewinnern" nach wenigen hundert Samples. Die Ursache ist fast immer dieselbe: Statistik wurde übersprungen.

Statistik gibt dir Werkzeuge, um Signal von Rauschen zu trennen. Sie zeigt dir, wann ein Unterschied real ist, wie sicher du sein solltest und wie viele Daten du brauchst, bevor du einem Ergebnis vertraust. Jede ML-Pipeline, jeder Modellvergleich und jedes Experiment braucht Statistik. Ohne sie rätst du nur.

## Das Konzept

### Deskriptive Statistik: Deine Daten zusammenfassen

Bevor du etwas modellierst, musst du verstehen, wie deine Daten aussehen. Deskriptive Statistik komprimiert einen Datensatz auf wenige Kennzahlen, die seine Form erfassen.

**Lageparameter** beantworten: „Wo liegt die Mitte?"

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

Der Mittelwert ist der Balancepunkt. Der Median ist die Mitte der sortierten Werte. Wenn beide stark auseinanderliegen, ist die Verteilung schief. Einkommensverteilungen haben oft Mittelwert >> Median (Rechtsschiefe durch sehr hohe Einkommen). Loss-Verteilungen im Training haben oft Mittelwert << Median (Linksschiefe durch viele einfache Samples).

**Streuungsmaße** beantworten: „Wie stark sind die Daten verteilt?"

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**Perzentile** teilen sortierte Daten in 100 gleich große Teile. Das 25. Perzentil (Q1) heißt: 25 % der Werte liegen darunter. Das 50. Perzentil ist der Median. Das 75. Perzentil ist Q3.

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

In ML sind Perzentile wichtig für Inferenzlatenz, Verteilungen von Vorhersagekonfidenzen und Fehlerverteilungen. Ein Modell mit kleinem Durchschnittsfehler, aber katastrophalem P99-Fehler kann in sicherheitskritischen Anwendungen unbrauchbar sein.

**Stichprobe vs. Grundgesamtheit.** Berechnest du die Varianz aus einer Stichprobe, teilst du durch (n-1) statt durch n. Das ist die Bessel-Korrektur. Sie gleicht aus, dass dein Stichprobenmittel nicht der wahre Populationsmittelwert ist. Mit n im Nenner unterschätzt du die echte Varianz systematisch. Mit (n-1) ist der Schätzer unverzerrt.

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

Praktisch gilt: Bei großem n (Tausende Samples) ist der Unterschied vernachlässigbar. Bei kleinem n (Dutzende Samples) ist er relevant.

### Korrelation: Wie Variablen gemeinsam variieren

Korrelation misst Stärke und Richtung einer linearen Beziehung zwischen zwei Variablen.

**Pearson-Korrelationskoeffizient** misst lineare Zusammenhänge:

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

Pearson setzt eine lineare Beziehung voraus und grob normalverteilte Variablen. Er ist empfindlich gegenüber Ausreißern. Ein einzelner extremer Punkt kann r von 0,1 auf 0,9 ziehen.

**Spearman-Rangkorrelation** misst monotone Zusammenhänge:

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**Wann welche verwenden:**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**Goldene Regel:** Korrelation impliziert keine Kausalität. Eisverkauf und Ertrinkungsfälle korrelieren, weil beides im Sommer steigt. Die Modellgenauigkeit und die Zahl der Parameter können korrelieren, aber mehr Parameter verbessern nicht automatisch die Genauigkeit (siehe Overfitting).

### Kovarianzmatrix

Die Kovarianz zweier Variablen misst, wie sie gemeinsam schwanken:

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

Für d Features ist die Kovarianzmatrix C eine d x d-Matrix, wobei C[i][j] = Cov(feature_i, feature_j). Die Diagonaleinträge C[i][i] sind die Varianzen der einzelnen Features.

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**Bezug zu PCA.** PCA zerlegt die Kovarianzmatrix per Eigenwertzerlegung. Die Eigenvektoren sind die Hauptkomponenten (Richtungen maximaler Varianz). Die Eigenwerte geben an, wie viel Varianz jede Komponente erklärt. Genau das war Lektion 10: Jetzt ist klar, warum die Kovarianzmatrix zerlegt wird – sie kodiert alle paarweisen linearen Beziehungen in deinen Daten.

**Bezug zur Korrelation.** Die Korrelationsmatrix ist die Kovarianzmatrix standardisierter Variablen (jede geteilt durch ihre Standardabweichung). Korrelation normalisiert die Kovarianz auf den Bereich [-1, 1].

### Hypothesentests

Hypothesentests sind ein Rahmen für Entscheidungen unter Unsicherheit. Du startest mit einer Behauptung, sammelst Daten und prüfst, ob die Daten mit dieser Behauptung vereinbar sind.

**Aufbau:**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**Der p-Wert** ist die Wahrscheinlichkeit, unter der Annahme von H0 Daten zu beobachten, die mindestens so extrem sind wie deine Beobachtung. Er ist **nicht** die Wahrscheinlichkeit, dass H0 wahr ist. Das ist das häufigste Missverständnis in der Statistik.

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**Konfidenzintervalle** geben einen plausiblen Wertebereich für einen Parameter an:

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

Die Breite des Intervalls zeigt die Präzision. Breite Intervalle bedeuten hohe Unsicherheit. Schmale Intervalle bedeuten präzise Schätzung (aber nicht automatisch korrekt, wenn die Daten verzerrt sind).

### Der t-Test

Der t-Test vergleicht Mittelwerte. Es gibt mehrere Varianten.

**Ein-Stichproben-t-Test:** Unterscheidet sich der Populationsmittelwert von einem vorgegebenen Wert?

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**Zwei-Stichproben-t-Test (unabhängig):** Unterscheiden sich die Mittelwerte zweier Gruppen?

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**Gepaarter t-Test:** Wenn Messungen paarweise vorliegen (z. B. gleiches Modell auf gleichen Datensplits):

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

Im ML ist der gepaarte t-Test häufig: Du testest beide Modelle auf denselben 10 Cross-Validation-Folds und vergleichst die Scores paarweise.

### Chi-Quadrat-Test

Der Chi-Quadrat-Test prüft, ob beobachtete Häufigkeiten zu erwarteten Häufigkeiten passen. Nützlich für kategoriale Daten.

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### A/B-Tests für ML-Modelle

A/B-Tests in ML sind nicht dasselbe wie Web-A/B-Tests. Beim Modellvergleich gibt es spezielle Herausforderungen:

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**Ablauf:**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### Statistische Signifikanz vs. praktische Signifikanz

Ein Ergebnis kann statistisch signifikant, aber praktisch bedeutungslos sein. Mit genügend Daten wird selbst ein winziger Unterschied statistisch signifikant.

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**Effektstärke** quantifiziert, wie groß ein Unterschied ist – unabhängig von der Stichprobengröße:

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

Berichte immer p-Wert und Effektstärke. Der p-Wert zeigt, ob der Unterschied real ist. Die Effektstärke zeigt, ob er wichtig ist.

### Problem multipler Vergleiche

Wenn du viele Hypothesen testest, sind einige rein zufällig „signifikant". Testest du 20 Dinge mit alpha = 0,05, erwartest du selbst ohne echten Effekt etwa 1 falsch-positives Ergebnis.

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Bonferroni-Korrektur:** Teile alpha durch die Anzahl der Tests.

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

In ML ist das wichtig, wenn du ein Modell über mehrere Metriken vergleichst, viele Hyperparameter-Konfigurationen testest oder auf mehreren Datensätzen evaluierst.

### Bootstrap-Methoden

Bootstrapping schätzt die Stichprobenverteilung einer Statistik, indem Daten mit Zurücklegen resampelt werden. Es sind keine Annahmen über die zugrunde liegende Verteilung nötig.

**Der Algorithmus:**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**Bootstrap-Konfidenzintervall (Perzentilmethode):**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**Warum Bootstrap für ML wichtig ist:**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**Bootstrap für Modellvergleich:**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

Das ist robuster als der gepaarte t-Test, weil keine Verteilungsannahmen gemacht werden.

### Parametrische vs. nichtparametrische Tests

**Parametrische Tests** setzen eine bestimmte Verteilung voraus (meist Normalverteilung):

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**Nichtparametrische Tests** machen keine Verteilungsannahmen:

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**Wann nichtparametrisch:**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**Wann parametrisch:**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

In ML-Experimenten ist n oft klein (5 oder 10 Cross-Validation-Folds), daher sind nichtparametrische Tests wie Wilcoxon Signed-Rank oft passender als t-Tests.

### Zentraler Grenzwertsatz: Praktische Implikationen

Der Zentrale Grenzwertsatz sagt: Die Verteilung von Stichprobenmitteln nähert sich mit wachsendem n einer Normalverteilung – unabhängig von der Verteilung der Grundgesamtheit.

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**Warum das für ML zählt:**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**Was der ZGS NICHT leistet:**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### Häufige Statistikfehler in ML-Papers

1. **Auf dem Trainingsset testen.** Garantiert Overfitting. Halte immer Daten zurück, die das Modell im Training nie sieht.

2. **Keine Konfidenzintervalle.** Eine einzelne Accuracy-Zahl ohne Unsicherheit macht Ergebnisse weder reproduzierbar noch überprüfbar.

3. **Multiple Vergleiche ignorieren.** 50 Konfigurationen testen und ohne Korrektur nur die beste berichten erhöht die Falsch-Positiv-Rate.

4. **Statistische und praktische Signifikanz verwechseln.** Ein p-Wert von 0,001 bei 0,01 % Accuracy-Gewinn ist nicht relevant.

5. **Accuracy bei unausgeglichenen Daten nutzen.** 99 % Accuracy bei 99 % Negativklasse heißt oft: Das Modell hat nichts gelernt. Nutze Precision, Recall, F1 oder AUC.

6. **Rosinenpicken bei Metriken.** Nur die Metrik berichten, in der dein Modell gewinnt. Saubere Evaluation berichtet alle relevanten Metriken.

7. **Informationsleck zwischen Train/Test-Splits.** Vor dem Split normalisieren oder zukünftige Daten zur Vorhersage der Vergangenheit nutzen.

8. **Kleine Testsets ohne Varianzschätzung.** Auf 100 Samples evaluieren und 2 % Verbesserung behaupten ist Rauschen, nicht Signal.

9. **Unabhängigkeit annehmen, obwohl Daten abhängig sind.** Medizinbilder desselben Patienten, mehrere Sätze aus demselben Dokument. Beobachtungen innerhalb einer Gruppe korrelieren.

10. **P-Hacking.** So lange verschiedene Tests, Teilmengen oder Ausschlusskriterien ausprobieren, bis p < 0,05 erreicht wird. Das Ergebnis ist ein Suchartefakt.

## Umsetzung

Du implementierst:

1. **Deskriptive Statistik von Grund auf** (Mittelwert, Median, Modus, Standardabweichung, Perzentile, IQR)
2. **Korrelationsfunktionen** (Pearson und Spearman, inklusive Kovarianzmatrix)
3. **Hypothesentests** (Ein-Stichproben-t-Test, Zwei-Stichproben-t-Test, Chi-Quadrat-Test)
4. **Bootstrap-Konfidenzintervalle** (für beliebige Statistiken, ohne Annahmen)
5. **A/B-Test-Simulator** (Daten generieren, testen, Fehler 1. und 2. Art prüfen)
6. **Demo: statistische vs. praktische Signifikanz** (zeigt, dass großes n fast alles „signifikant" macht)

Alles von Grund auf, nur mit `math` und `random`. Kein numpy, kein scipy.

## Schlüsselbegriffe

| Begriff | Definition |
|---|---|
| Mittelwert | Summe der Werte geteilt durch ihre Anzahl. Empfindlich gegenüber Ausreißern. |
| Median | Mittlerer Wert sortierter Daten. Robust gegenüber Ausreißern. |
| Standardabweichung | Quadratwurzel der Varianz. Misst die Streuung in Originaleinheiten. |
| Perzentil | Wert, unter dem ein bestimmter Prozentsatz der Daten liegt. |
| IQR | Interquartilsabstand. Q3 minus Q1. Die Streuung der mittleren 50 %. |
| Pearson-Korrelation | Misst lineare Zusammenhänge zwischen zwei Variablen. Bereich [-1, 1]. |
| Spearman-Korrelation | Misst monotone Zusammenhänge anhand von Rängen. |
| Kovarianzmatrix | Matrix der paarweisen Kovarianzen zwischen allen Features. |
| Nullhypothese | Standardannahme: kein Effekt oder kein Unterschied. |
| p-Wert | Wahrscheinlichkeit für mindestens so extreme Daten unter der Annahme, dass die Nullhypothese wahr ist. |
| Konfidenzintervall | Bereich plausibler Werte eines Parameters bei gegebenem Konfidenzniveau. |
| t-Test | Testet, ob sich Mittelwerte signifikant unterscheiden. Nutzt die t-Verteilung. |
| Chi-Quadrat-Test | Testet, ob beobachtete Häufigkeiten von erwarteten Häufigkeiten abweichen. |
| Effektstärke | Größe eines Unterschieds, unabhängig von der Stichprobengröße. Häufig: Cohen's d. |
| Bonferroni-Korrektur | Teilt die Signifikanzschwelle durch die Zahl der Tests, um Falsch-Positive zu kontrollieren. |
| Bootstrap | Resampling mit Zurücklegen zur Schätzung von Stichprobenverteilungen. |
| Fehler 1. Art | Falsch-positiv. H0 ablehnen, obwohl sie wahr ist. |
| Fehler 2. Art | Falsch-negativ. H0 nicht ablehnen, obwohl sie falsch ist. |
| Teststärke (Power) | Wahrscheinlichkeit, eine falsche H0 korrekt abzulehnen. Power = 1 minus Fehler-2-Rate. |
| Zentraler Grenzwertsatz | Stichprobenmittel konvergieren mit wachsender Stichprobengröße zu einer Normalverteilung. |
| Parametrischer Test | Nimmt eine bestimmte Datenverteilung an (meist normalverteilt). |
| Nichtparametrischer Test | Macht keine Verteilungsannahmen. Arbeitet mit Rängen oder Vorzeichen. |
