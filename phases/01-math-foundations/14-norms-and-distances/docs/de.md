# Normen und Distanzen

> Deine Distanzfunktion legt fest, was „ähnlich" bedeutet. Wählst du falsch, bricht alles danach auseinander.

**Typ:** Build
**Sprache:** Python
**Voraussetzungen:** Phase 1, Lektionen 01 (Lineare-Algebra-Intuition), 02 (Vektoren, Matrizen & Operationen)
**Dauer:** ~90 Minuten

## Lernziele

- L1-, L2-, Kosinus-, Mahalanobis-, Jaccard- und Edit-Distanzfunktionen von Grund auf implementieren
- Für eine gegebene ML-Aufgabe die passende Distanzmetrik auswählen und erklären, warum Alternativen scheitern
- L1- und L2-Normen mit LASSO- und Ridge-Regularisierung sowie ihren geometrischen Nebenbedingungsbereichen verknüpfen
- Zeigen, wie derselbe Datensatz unter verschiedenen Metriken unterschiedliche nächste Nachbarn ergibt

## Das Problem

Du hast zwei Vektoren. Vielleicht sind es Word-Embeddings. Vielleicht Nutzerprofile. Vielleicht Pixel-Arrays. Du musst wissen: Wie nah sind sie?

Die Antwort hängt vollständig davon ab, welche Distanzfunktion du wählst. Zwei Datenpunkte können unter einer Metrik nächste Nachbarn sein und unter einer anderen weit auseinanderliegen. Dein KNN-Klassifikator, deine Empfehlungs-Engine, deine Vektordatenbank, dein Clustering-Algorithmus, deine Loss-Funktion – sie alle hängen von dieser Wahl ab. Triffst du die falsche, optimiert dein Modell auf das Falsche.

Es gibt keine universell beste Distanz. L2 funktioniert gut für räumliche Daten. Kosinus-Ähnlichkeit dominiert NLP. Jaccard behandelt Mengen. Edit-Distanz behandelt Strings. Mahalanobis berücksichtigt Korrelationen. Wasserstein verschiebt Wahrscheinlichkeitsmasse. Jede davon kodiert eine andere Annahme darüber, was „ähnlich" bedeutet.

Diese Lektion baut jede wichtige Distanzfunktion von Grund auf, zeigt dir, wann welche das richtige Werkzeug ist, und demonstriert, wie dieselben Daten je nach Metrik völlig unterschiedliche nächste Nachbarn liefern.

## Das Konzept

### Normen: Vektorgröße messen

Eine Norm misst die „Größe" eines Vektors. Jede Distanzfunktion zwischen zwei Vektoren kann als Norm ihrer Differenz geschrieben werden: d(a, b) = ||a - b||. Normen zu verstehen heißt also, Distanzen zu verstehen.

### L1-Norm (Manhattan-Distanz)

Die L1-Norm summiert die Absolutwerte aller Komponenten.

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

Sie heißt Manhattan-Distanz, weil sie misst, wie weit du in einem Stadtgitter läufst, in dem du dich nur entlang der Achsen bewegen kannst. Keine Diagonalen.

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

Wann L1 verwenden:
- Hochdimensionale, spärliche Daten (Text-Features, One-Hot-Encoding)
- Wenn du Robustheit gegenüber Ausreißern willst (eine einzelne große Abweichung dominiert nicht)
- Feature-Selection-Probleme (L1-Regularisierung fördert Sparsity)

Verbindung zur L1-Regularisierung (Lasso): Das Hinzufügen von ||w||_1 zur Loss-Funktion bestraft die Summe absoluter Gewichtswerte. Dadurch werden kleine Gewichte auf exakt null gedrückt und automatische Feature-Selektion durchgeführt. Die L1-Strafe erzeugt diamantförmige Nebenbedingungsbereiche im Gewichtsraum, und die Ecken der Diamanten liegen auf den Achsen, wo manche Gewichte null sind.

Verbindung zu Loss-Funktionen: Mean Absolute Error (MAE) ist die durchschnittliche L1-Distanz zwischen Vorhersagen und Zielwerten. Sie bestraft alle Fehler linear und ist im Vergleich zu MSE robuster gegenüber Ausreißern.

### L2-Norm (Euklidische Distanz)

Die L2-Norm ist die Luftlinien-Distanz. Quadratwurzel aus der Summe der quadrierten Komponenten.

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

Das ist die Distanz, die du im Geometrieunterricht gelernt hast. Pythagoras in n Dimensionen.

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

Wann L2 verwenden:
- Kontinuierliche Daten mit niedriger bis mittlerer Dimension
- Wenn die Feature-Skalen vergleichbar sind
- Physische Distanzen (räumliche Daten, Sensorsignale)
- Bildähnlichkeit auf Pixelebene

Verbindung zur L2-Regularisierung (Ridge): Das Hinzufügen von ||w||_2^2 zur Loss-Funktion bestraft große Gewichte. Anders als L1 drückt sie Gewichte nicht auf null. Stattdessen schrumpft sie alle Gewichte proportional Richtung null. Die L2-Strafe erzeugt kreisförmige Nebenbedingungsbereiche, daher gibt es keine Ecken auf den Achsen. Gewichte werden klein, aber selten exakt null.

Verbindung zu Loss-Funktionen: Mean Squared Error (MSE) ist der Durchschnitt quadrierter L2-Distanzen. Das Quadrieren bestraft große Fehler stärker als kleine.

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp-Normen: die allgemeine Familie

L1 und L2 sind Spezialfälle der Lp-Norm:

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

Verschiedene Werte von p erzeugen unterschiedlich geformte „Einheitskugeln" (die Menge aller Punkte mit Distanz 1 zum Ursprung):

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-Unendlich-Norm (Chebyshev-Distanz)

Wenn p gegen unendlich geht, konvergiert die Lp-Norm zur maximalen absoluten Komponente.

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

Die Distanz zwischen zwei Punkten wird durch die eine Dimension bestimmt, in der sie sich am stärksten unterscheiden. Alle anderen Dimensionen werden ignoriert.

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

Wann L-Unendlich verwenden:
- Wenn die Worst-Case-Abweichung in einer einzelnen Dimension zählt
- Spielbretter (ein König im Schach bewegt sich in L-Unendlich: ein Schritt in jede Richtung kostet 1)
- Fertigungstoleranzen (jede Dimension muss in der Spezifikation liegen)

### Kosinus-Ähnlichkeit und Kosinus-Distanz

Die Kosinus-Ähnlichkeit misst den Winkel zwischen zwei Vektoren und ignoriert ihre Beträge.

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

Sie reicht von -1 (entgegengesetzte Richtung) bis +1 (gleiche Richtung). Senkrechte Vektoren haben Kosinus-Ähnlichkeit 0.

Kosinus-Distanz wandelt sie in eine Distanz um: cosine_distance = 1 - cosine_similarity. Das reicht von 0 (identische Richtung) bis 2 (entgegengesetzte Richtung).

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

Warum Kosinus NLP und Embeddings dominiert: In Texten sollte die Dokumentlänge die Ähnlichkeit nicht beeinflussen. Ein Dokument über Katzen, das doppelt so lang ist wie ein anderes über Katzen, sollte weiterhin „ähnlich" sein. Kosinus-Ähnlichkeit ignoriert den Betrag (Länge) und betrachtet nur die Richtung. Zwei Dokumente mit gleicher Wortverteilung, aber unterschiedlicher Länge, zeigen in dieselbe Richtung und erhalten Kosinus-Ähnlichkeit 1.0.

Wann Kosinus-Ähnlichkeit verwenden:
- Textähnlichkeit (TF-IDF-Vektoren, Word-Embeddings, Satz-Embeddings)
- Domänen, in denen Betrag Rauschen und Richtung Signal ist
- Empfehlungssysteme (Nutzerpräferenz-Vektoren)
- Embedding-Suche (Vektordatenbanken verwenden fast immer Kosinus oder Skalarprodukt)

### Skalarprodukt-Ähnlichkeit vs. Kosinus-Ähnlichkeit

Das Skalarprodukt zweier Vektoren ist:

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

Kosinus-Ähnlichkeit ist das durch beide Beträge normalisierte Skalarprodukt. Wenn beide Vektoren bereits auf Einheitslänge normiert sind (Betrag = 1), sind Skalarprodukt und Kosinus-Ähnlichkeit identisch.

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

Wann sie sich unterscheiden: Das Skalarprodukt enthält Betragsinformation. Ein Vektor mit größerem Betrag erhält einen höheren Skalarprodukt-Score. Das ist in einigen Retrieval-Systemen wichtig, wenn „populäre" Elemente höher gerankt werden sollen. Der Betrag wirkt dann als implizites Qualitäts- oder Wichtigkeitssignal.

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

In der Praxis:
- Verwende Kosinus-Ähnlichkeit, wenn du reine Richtungsähnlichkeit willst
- Verwende Skalarprodukt, wenn Beträge sinnvolle Information tragen
- Viele Vektordatenbanken (Pinecone, Weaviate, Qdrant) lassen dich zwischen beiden wählen
- Wenn deine Embeddings L2-normalisiert sind, spielt die Wahl keine Rolle

### Mahalanobis-Distanz

Die euklidische Distanz behandelt alle Dimensionen gleich. Sind deine Features aber korreliert oder unterschiedlich skaliert, liefert L2 irreführende Ergebnisse.

Die Mahalanobis-Distanz berücksichtigt die Kovarianzstruktur der Daten.

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

wobei S die Kovarianzmatrix der Daten ist.

Intuitiv: Die Mahalanobis-Distanz dekorreliert und normalisiert die Daten zuerst (Whitening) und berechnet dann die L2-Distanz in diesem transformierten Raum. Ist S die Einheitsmatrix (unkorrelierte Features mit Einheitsvarianz), reduziert sich die Mahalanobis-Distanz auf die euklidische Distanz.

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

Wann Mahalanobis-Distanz verwenden:
- Ausreißererkennung (Punkte mit großer Mahalanobis-Distanz zum Mittelwert sind Ausreißer)
- Klassifikation, wenn Features unterschiedliche Skalen und Korrelationen haben
- Wenn du genug Daten hast, um eine verlässliche Kovarianzmatrix zu schätzen
- Qualitätskontrolle in der Fertigung (multivariate Prozessüberwachung)

### Jaccard-Ähnlichkeit (für Mengen)

Jaccard-Ähnlichkeit misst die Überlappung zwischen zwei Mengen.

```
J(A, B) = |A intersect B| / |A union B|
```

Sie reicht von 0 (keine Überlappung) bis 1 (identische Mengen). Jaccard-Distanz = 1 - Jaccard-Ähnlichkeit.

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

Wann Jaccard verwenden:
- Vergleich von Mengen aus Tags, Kategorien oder Features
- Dokumentähnlichkeit basierend auf Wortvorkommen (nicht Häufigkeit)
- Near-Duplicate-Erkennung (MinHash-Approximation von Jaccard)
- Vergleich binärer Feature-Vektoren (Vorhanden/Nicht-vorhanden-Daten)
- Bewertung von Segmentierungsmodellen (Intersection over Union = Jaccard)

### Edit-Distanz (Levenshtein-Distanz)

Edit-Distanz zählt die minimale Anzahl von Operationen auf Zeichenebene, die nötig sind, um einen String in einen anderen zu überführen. Die Operationen sind: Einfügen, Löschen oder Ersetzen.

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

Berechnet wird sie mit dynamischer Programmierung. Man füllt eine Matrix, in der der Eintrag (i, j) die Edit-Distanz zwischen den ersten i Zeichen von String A und den ersten j Zeichen von String B ist.

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

Wann Edit-Distanz verwenden:
- Rechtschreibprüfung und -korrektur
- DNA-Sequenz-Alignment (mit gewichteten Operationen)
- Fuzzy-String-Matching
- Deduplizierung unstrukturierter Textdaten

### KL-Divergenz (keine Distanz, wird aber oft so genutzt)

KL-Divergenz misst, wie stark sich eine Wahrscheinlichkeitsverteilung von einer anderen unterscheidet. Sie wurde in Lektion 09 behandelt, gehört aber in diese Diskussion, weil sie trotz fehlender Distanz-Eigenschaften oft als „Distanz" verwendet wird.

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

Kritische Eigenschaft: KL-Divergenz ist NICHT symmetrisch.

```
D_KL(P || Q) != D_KL(Q || P)
```

Damit verletzt sie die Grundanforderung an eine Distanzmetrik. Außerdem erfüllt sie die Dreiecksungleichung nicht. Sie ist eine Divergenz, keine Distanz.

Forward KL (D_KL(P || Q)) ist „mean-seeking": Q versucht, alle Modi von P abzudecken.
Reverse KL (D_KL(Q || P)) ist „mode-seeking": Q konzentriert sich auf einen einzelnen Modus von P.

Wenn du KL-Divergenz siehst:
- VAEs (der KL-Term in der ELBO drückt die latente Verteilung in Richtung Prior)
- Knowledge Distillation (das Student-Modell versucht, die Verteilung des Teacher-Modells zu treffen)
- RLHF (die KL-Strafe hält das feinabgestimmte Modell nah am Basismodell)
- Policy-Gradient-Methoden (Einschränkung von Policy-Updates)

### Wasserstein-Distanz (Earth Mover's Distance)

Wasserstein-Distanz misst die minimale „Arbeit", die nötig ist, um eine Wahrscheinlichkeitsverteilung in eine andere zu überführen. Stell dir vor: Eine Verteilung ist ein Erdhaufen und die andere ein Loch – wie viel Erde musst du wie weit bewegen?

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

Für 1D-Verteilungen vereinfacht sich das zur Fläche der absoluten Differenz der kumulativen Verteilungsfunktionen:

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

Warum Wasserstein wichtig ist:
- Sie ist eine echte Metrik (symmetrisch, erfüllt die Dreiecksungleichung)
- Sie liefert Gradienten, selbst wenn Verteilungen nicht überlappen (KL-Divergenz geht gegen unendlich)
- Diese Eigenschaft machte sie zentral für Wasserstein-GANs (WGANs), die die Trainingsinstabilität ursprünglicher GANs lösten

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

Wann Wasserstein verwenden:
- GAN-Training (WGAN, WGAN-GP)
- Vergleich von Verteilungen, die sich eventuell nicht überlappen
- Optimal-Transport-Probleme
- Bild-Retrieval (Vergleich von Farbhistogrammen)

### Warum verschiedene Aufgaben verschiedene Distanzen brauchen

| Aufgabe | Beste Distanz | Warum |
|------|--------------|-----|
| Textähnlichkeit | Kosinus | Betrag ist Rauschen, Richtung ist Bedeutung |
| Bildvergleich auf Pixelebene | L2 | Räumliche Beziehungen sind wichtig, Features haben vergleichbare Skala |
| Spärliche hochdimensionale Features | L1 | Robust, verstärkt seltene große Unterschiede nicht |
| Mengenüberlappung (Tags, Kategorien) | Jaccard | Daten sind natürlich mengenwertig, nicht vektoriell |
| String-Matching | Edit-Distanz | Operationen entsprechen menschlicher Bearbeitungsintuition |
| Ausreißererkennung | Mahalanobis | Berücksichtigt Feature-Korrelationen und -Skalen |
| Verteilungen vergleichen | KL-Divergenz | Misst Informationsverlust bei Nutzung von Q statt P |
| GAN-Training | Wasserstein | Liefert Gradienten, selbst wenn Verteilungen nicht überlappen |
| Embeddings (Vektor-DB) | Kosinus oder Skalarprodukt | Embeddings werden trainiert, Bedeutung in Richtung zu kodieren |
| Empfehlung | Skalarprodukt | Betrag kann Popularität oder Vertrauen kodieren |
| DNA-Sequenzen | Gewichtete Edit-Distanz | Substitutionskosten hängen vom Nukleotidpaar ab |
| Fertigungs-QC | L-Unendlich | Worst-Case-Abweichung in einer Dimension ist entscheidend |

### Verbindung zu Loss-Funktionen

Loss-Funktionen sind Distanzfunktionen, angewendet auf Vorhersagen gegenüber Zielwerten.

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### Verbindung zu Regularisierung

Regularisierung fügt der Loss-Funktion eine Normstrafe auf die Gewichte hinzu.

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

Warum L1 Sparsity erzeugt und L2 nicht: Stell dir den Nebenbedingungsbereich im 2D-Gewichtsraum vor. L1 ist ein Diamant, L2 ein Kreis. Die Konturen der Loss-Funktion (Ellipsen) berühren den Diamanten am ehesten in einer Ecke, wo ein Gewicht null ist. Den Kreis berühren sie in einem glatten Punkt, wo beide Gewichte ungleich null sind.

### Suche nach nächsten Nachbarn

Jede Distanzfunktion impliziert ein Nearest-Neighbor-Suchproblem: Gegeben ein Query-Punkt, finde die nächsten Punkte in einem Datensatz.

Exakte Nearest-Neighbor-Suche ist O(n * d) pro Query in einem Datensatz mit n Punkten und d Dimensionen. Für große Datensätze ist das zu langsam.

Approximate Nearest Neighbor (ANN)-Algorithmen tauschen ein wenig Genauigkeit gegen massive Geschwindigkeitsgewinne:

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW (Hierarchical Navigable Small World) ist der dominante Algorithmus in modernen Vektordatenbanken. Er baut einen mehrschichtigen Graphen auf, in dem jeder Knoten mit seinen ungefähren nächsten Nachbarn verbunden ist. Die Suche startet in der obersten Schicht (dünn, große Sprünge) und steigt bis zur untersten Schicht ab (dicht, kurze Sprünge).

## Baue es

### Schritt 1: Alle Norm- und Distanzfunktionen

Siehe `code/distances.py` für die vollständige Implementierung. Jede Funktion wird von Grund auf nur mit grundlegender Python-Mathematik gebaut.

### Schritt 2: Gleiche Daten, verschiedene Distanzen, verschiedene Nachbarn

Die Demo in `distances.py` erstellt einen Datensatz, wählt einen Query-Punkt und zeigt, wie sich der nächste Nachbar je nach Distanzmetrik ändert. Der Punkt, der unter L1 „am nächsten" ist, muss unter L2 oder Kosinus nicht am nächsten sein.

### Schritt 3: Embedding-Ähnlichkeitssuche

Der Code enthält eine simulierte Embedding-Ähnlichkeitssuche, die mit Kosinus-Ähnlichkeit vs. L2-Distanz die ähnlichsten „Dokumente" zu einer Query findet und zeigt, dass sich die Rankings unterscheiden können.

## Nutze es

Die häufigste praktische Anwendung: ähnliche Elemente in einer Vektordatenbank finden.

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

Wenn du `model.encode(text)` aufrufst und dann in einer Vektordatenbank suchst, passiert im Kern genau das. Das Embedding-Modell bildet Text auf Vektoren ab. Die Vektordatenbank berechnet Kosinus-Ähnlichkeit (oder Skalarprodukt) zwischen deinem Query-Vektor und jedem gespeicherten Vektor und nutzt ANN-Algorithmen, um nicht alle prüfen zu müssen.

## Übungen

1. Berechne L1-, L2- und L-Unendlich-Distanzen zwischen (1, 2, 3) und (4, 0, 6). Verifiziere, dass immer L-inf <= L2 <= L1 für jedes Punktepaar gilt. Beweise, warum diese Ordnung garantiert ist.

2. Erstelle zwei Vektoren, bei denen die Kosinus-Ähnlichkeit hoch ist (> 0.9), aber die L2-Distanz groß ist (> 10). Erkläre geometrisch, was passiert. Erstelle dann zwei Vektoren, bei denen die Kosinus-Ähnlichkeit niedrig ist (< 0.3), aber die L2-Distanz klein ist (< 0.5).

3. Implementiere eine Funktion, die einen Datensatz und einen Query-Punkt nimmt und den nächsten Nachbarn unter L1-, L2-, Kosinus- und Mahalanobis-Distanz zurückgibt. Finde einen Datensatz, bei dem sich alle vier uneinig sind, welcher Punkt der nächste ist.

4. Berechne die Wasserstein-Distanz zwischen [0.5, 0.5, 0, 0] und [0, 0, 0.5, 0.5] von Hand mit der CDF-Methode. Berechne sie dann zwischen [0.25, 0.25, 0.25, 0.25] und [0, 0, 0.5, 0.5]. Welche ist größer und warum?

5. Implementiere MinHash für approximative Jaccard-Ähnlichkeit. Erzeuge 100 Zufallsmengen, berechne exakte Jaccard-Werte für alle Paare und vergleiche sie mit der MinHash-Approximation bei 50, 100 und 200 Hashfunktionen. Plotte den Approximationsfehler.

## Schlüsselbegriffe

| Begriff | Was man oft sagt | Was es tatsächlich bedeutet |
|------|----------------|----------------------|
| Norm | „Größe eines Vektors" | Eine Funktion, die einen Vektor auf einen nichtnegativen Skalar abbildet und Dreiecksungleichung, absolute Homogenität sowie Null nur für den Nullvektor erfüllt |
| L1-Norm | „Manhattan-Distanz" | Summe absoluter Komponentenwerte. Erzeugt Sparsity in der Optimierung. Robust gegen Ausreißer |
| L2-Norm | „Euklidische Distanz" | Quadratwurzel der Summe quadrierter Komponenten. Die Luftlinien-Distanz im euklidischen Raum |
| Lp-Norm | „Verallgemeinerte Norm" | p-te Wurzel der Summe p-ter Potenzen absoluter Komponenten. L1 und L2 sind Spezialfälle |
| L-Unendlich-Norm | „Max-Norm" oder „Chebyshev-Distanz" | Der maximale absolute Komponentenwert. Der Grenzfall der Lp-Norm für p gegen unendlich |
| Kosinus-Ähnlichkeit | „Winkel zwischen Vektoren" | Durch beide Beträge normalisiertes Skalarprodukt. Bereich von -1 bis +1. Ignoriert Vektorlänge |
| Kosinus-Distanz | „1 minus Kosinus-Ähnlichkeit" | Wandelt Kosinus-Ähnlichkeit in eine Distanz um. Bereich von 0 bis 2 |
| Skalarprodukt | „Unnormalisierter Kosinus" | Summe komponentenweiser Produkte. Entspricht Kosinus-Ähnlichkeit mal beiden Beträgen |
| Mahalanobis-Distanz | „Korrelation-bewusste Distanz" | L2-Distanz in einem per Datenkovarianzmatrix gewichteten (dekorrelierten und normalisierten) Raum |
| Jaccard-Ähnlichkeit | „Mengenüberlappung" | Größe der Schnittmenge geteilt durch Größe der Vereinigungsmenge. Für Mengen, nicht Vektoren |
| Edit-Distanz | „Levenshtein-Distanz" | Minimale Einfügungen, Löschungen und Ersetzungen, um einen String in einen anderen zu überführen |
| KL-Divergenz | „Distanz zwischen Verteilungen" | Keine echte Distanz (nicht symmetrisch). Misst zusätzliche Bits, wenn Q zum Kodieren von P genutzt wird |
| Wasserstein-Distanz | „Earth Mover's Distance" | Minimale Arbeit, um Masse von einer Verteilung in eine andere zu transportieren. Eine echte Metrik |
| Approximate Nearest Neighbor | „ANN-Suche" | Algorithmen (HNSW, LSH, IVF), die annähernd nächste Punkte viel schneller als exakt finden |
| HNSW | „Der Vektor-DB-Algorithmus" | Hierarchical Navigable Small World Graph. Mehrschichtiger Graph für schnelle approximative Nearest-Neighbor-Suche |
| L1-Regularisierung | „Lasso" | Hinzufügen der L1-Norm der Gewichte zur Loss-Funktion. Drückt Gewichte auf null (Sparsity) |
| L2-Regularisierung | „Ridge" oder „Weight Decay" | Hinzufügen der quadrierten L2-Norm der Gewichte zur Loss-Funktion. Schrumpft Gewichte Richtung null ohne Sparsity |
| Elastic Net | „L1 + L2" | Kombiniert L1- und L2-Regularisierung. Geht besser mit korrelierten Feature-Gruppen um als jede allein |

## Weiterführende Literatur

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - Metas Bibliothek für ANN-Suche im Milliardenmaßstab
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - das Paper, das die Earth Mover's Distance in GANs einführte
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - grundlegender ANN-Algorithmus
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec, wo Kosinus-Ähnlichkeit zum Standard für Embeddings wurde
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - praktischer Leitfaden zu Distanzmetriken und Neighbor-Algorithmen in scikit-learn
