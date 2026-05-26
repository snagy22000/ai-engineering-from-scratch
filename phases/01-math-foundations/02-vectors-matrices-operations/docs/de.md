# Vektoren, Matrizen & Operationen

> Jedes neuronale Netz ist im Grunde nur Matrizenmultiplikation mit ein paar extra Schritten.

**Typ:** Aufbauen
**Sprachen:** Python, Julia
**Voraussetzungen:** Phase 1, Lektion 01 (Lineare Algebra – Intuition)
**Zeit:** ~60 Minuten

## Lernziele

- Eine Matrix-Klasse mit elementweisen Operationen, Matrizenmultiplikation, Transponierung, Determinante und Inverser aufbauen
- Elementweise Multiplikation von Matrizenmultiplikation unterscheiden und erklären, wann welche angewendet wird
- Eine einzelne dichte Schicht eines neuronalen Netzes (`relu(W @ x + b)`) nur mit der selbst geschriebenen Matrix-Klasse implementieren
- Broadcasting-Regeln erklären und zeigen, wie die Bias-Addition in neuronalen Netzwerk-Frameworks funktioniert

## Das Problem

Du möchtest ein neuronales Netz bauen. Du liest den Code und siehst:

```
output = activation(weights @ input + bias)
```

Das `@` ist Matrizenmultiplikation. `weights` ist eine Matrix. `input` ist ein Vektor. Wenn du nicht weißt, was diese Operationen tun, ist diese Zeile Magie. Wenn du es weißt, ist es der gesamte Vorwärtsdurchlauf einer Schicht in drei Operationen.

Jedes Bild, das dein Modell verarbeitet, ist eine Matrix aus Pixelwerten. Jedes Word-Embedding ist ein Vektor. Jede Schicht jedes neuronalen Netzes ist eine Matrizentransformation. Du kannst keine KI-Systeme bauen, ohne in Matrizenoperationen fließend zu sein – genauso wenig wie du Code schreiben kannst, ohne Variablen zu verstehen.

Diese Lektion baut diese Fähigkeit von Grund auf.

## Das Konzept

### Vektoren: geordnete Zahlenlisten

Ein Vektor ist eine Liste von Zahlen mit einer Richtung und einem Betrag. In der KI repräsentieren Vektoren Datenpunkte, Features oder Parameter.

```
v = [3, 4]        -- ein 2D-Vektor
w = [1, 0, -2]    -- ein 3D-Vektor
```

Ein 2D-Vektor `[3, 4]` zeigt auf die Koordinaten (3, 4) in einer Ebene. Seine Länge (Betrag) ist 5 (das 3-4-5-Dreieck).

### Matrizen: Zahlenraster

Eine Matrix ist ein 2D-Raster. Zeilen und Spalten. Eine m × n-Matrix hat m Zeilen und n Spalten.

```
A = | 1  2  3 |     -- 2×3-Matrix (2 Zeilen, 3 Spalten)
    | 4  5  6 |
```

In neuronalen Netzen transformieren Gewichtsmatrizen Eingabevektoren in Ausgabevektoren. Eine Schicht mit 784 Eingaben und 128 Ausgaben verwendet eine 128×784-Gewichtsmatrix.

### Warum Formen wichtig sind

Matrizenmultiplikation hat eine strenge Regel: `(m × n) @ (n × p) = (m × p)`. Die inneren Dimensionen müssen übereinstimmen.

```
(128 × 784) @ (784 × 1) = (128 × 1)
  Gewichte      Eingabe    Ausgabe

Innere Dimensionen: 784 = 784  -- gültig
```

Wenn du in PyTorch einen Shape-Mismatch-Fehler bekommst, liegt das hier.

### Die Operationsübersicht

| Operation | Was sie tut | Verwendung in neuronalen Netzen |
|-----------|------------|--------------------------------|
| Addition | Elementweise kombinieren | Bias zur Ausgabe addieren |
| Skalarmultiplikation | Jedes Element skalieren | Lernrate * Gradienten |
| Matrizenmultiplikation | Vektoren transformieren | Vorwärtsdurchlauf einer Schicht |
| Transponierung | Zeilen und Spalten tauschen | Rückwärtsdurchlauf (Backpropagation) |
| Determinante | Zusammenfassende Kennzahl | Invertierbarkeit prüfen |
| Inverse | Transformation rückgängig machen | Lineare Systeme lösen |
| Einheitsmatrix | Nichts-tun-Matrix | Initialisierung, Residual-Verbindungen |

### Elementweise vs. Matrizenmultiplikation

Diese Unterscheidung verwirrt Anfänger ständig.

Elementweise: entsprechende Positionen multiplizieren. Beide Matrizen müssen die gleiche Form haben.

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

Matrizenmultiplikation: Skalarprodukte aus Zeilen und Spalten. Innere Dimensionen müssen übereinstimmen.

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

Verschiedene Operationen, verschiedene Ergebnisse, verschiedene Regeln.

### Broadcasting

Wenn man einen Bias-Vektor zu einer Matrix von Ausgaben addiert, stimmen die Formen nicht überein. Broadcasting dehnt das kleinere Array aus, um es anzupassen.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting dehnt den Vektor über die Zeilen aus:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

Jedes moderne Framework macht das automatisch. Das zu verstehen verhindert Verwirrung, wenn Formen scheinbar falsch sind, der Code aber läuft.

## Umsetzung

### Schritt 1: Vektor-Klasse

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### Schritt 2: Matrix-Klasse mit Kernoperationen

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix ist singulär, keine Inverse vorhanden")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### Schritt 3: Anwendung ansehen

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### Schritt 4: Verbindung zu neuronalen Netzen

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Eingabe-Form: {inputs.shape}")
print(f"Gewichts-Form: {weights.shape}")
print(f"Ausgabe-Form: {output.shape}")
print(f"Ausgabe: {output.data}")
```

Das ist eine einzelne dichte Schicht: `output = relu(W @ x + b)`. Jede dichte Schicht in jedem neuronalen Netz tut genau das.

## In der Praxis

NumPy erledigt alles oben in weniger Zeilen und um Größenordnungen schneller.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (elementweise) =\n", A * B)
print("A @ B (Matrizenmultiplikation) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeuronale Netzschicht: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Ausgabe:\n{output}")
```

Der `@`-Operator in Python ruft `__matmul__` auf. NumPy implementiert es mit optimierten BLAS-Routinen in C und Fortran. Gleiche Mathematik, 100× schneller.

Broadcasting in NumPy:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy broadcasted den 1D-Bias automatisch über beide Zeilen. So funktioniert Bias-Addition in jedem neuronalen Netzwerk-Framework.

## Fertigstellen

Diese Lektion erzeugt einen Prompt zum Lehren von Matrizenoperationen durch geometrische Intuition. Siehe `outputs/prompt-matrix-operations.md`.

Die hier gebaute Matrix-Klasse ist die Grundlage für das Mini-Neural-Network-Framework, das wir in Phase 3, Lektion 10 bauen.

## Übungen

1. **Inverse überprüfen.** Multipliziere `A @ A.inverse_2x2()` und bestätige, dass du die Einheitsmatrix erhältst. Probiere es mit drei verschiedenen 2×2-Matrizen. Was passiert, wenn die Determinante null ist?

2. **3×3-Inverse implementieren.** Erweitere die Matrix-Klasse, um Inverse für 3×3-Matrizen mit der Adjunkten-Methode zu berechnen. Teste es gegen NumPys `np.linalg.inv`.

3. **Ein zweischichtiges Netz bauen.** Nur mit deiner Matrix-Klasse (kein NumPy): erstelle ein zweischichtiges neuronales Netz: Eingabe (3) → Versteckt (4) → Ausgabe (2). Initialisiere zufällige Gewichte, führe einen Vorwärtsdurchlauf durch und überprüfe, ob alle Formen korrekt sind.

## Schlüsselbegriffe

| Begriff | Was man sagt | Was es wirklich bedeutet |
|---------|-------------|--------------------------|
| Vektor | „Ein Pfeil" | Eine geordnete Liste von Zahlen. In der KI: ein Punkt im hochdimensionalen Raum. |
| Matrix | „Eine Zahlentabelle" | Eine lineare Transformation. Sie bildet Vektoren von einem Raum auf einen anderen ab. |
| Matrizenmultiplikation | „Einfach die Zahlen multiplizieren" | Skalarprodukte zwischen jeder Zeile der ersten Matrix und jeder Spalte der zweiten. Reihenfolge ist wichtig. |
| Transponierung | „Umdrehen" | Zeilen und Spalten tauschen. Wandelt eine m×n-Matrix in n×m um. Entscheidend beim Backpropagation. |
| Determinante | „Eine Zahl aus der Matrix" | Misst, wie viel die Matrix Fläche (2D) oder Volumen (3D) skaliert. Null bedeutet, die Transformation quetscht eine Dimension. |
| Inverse | „Die Matrix rückgängig machen" | Die Matrix, die die Transformation umkehrt. Existiert nur, wenn die Determinante nicht null ist. |
| Einheitsmatrix | „Die langweilige Matrix" | Das Matrix-Äquivalent der Multiplikation mit 1. Verwendet in Residual-Verbindungen (ResNets). |
| Broadcasting | „Magische Form-Anpassung" | Ein kleineres Array auf ein größeres ausdehnen, indem es entlang fehlender Dimensionen wiederholt wird. |
| Elementweise | „Normale Multiplikation" | Entsprechende Positionen multiplizieren. Beide Arrays müssen die gleiche Form haben (oder broadcasted werden können). |

## Weiterführende Literatur

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) – visuelle Intuition für jede hier behandelte Operation
- [NumPy-Dokumentation zu Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) – die genauen Regeln, die NumPy befolgt
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) – präzise Referenz für ML-spezifische lineare Algebra
