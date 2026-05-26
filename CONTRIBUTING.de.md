# Mitwirken

Lektionen, Übersetzungen, Fixes und Outputs sind willkommen. Eine Änderung pro Pull Request hält Reviews schnell und nachvollziehbar.

## Wichtig: README und ROADMAP speisen die Website

`site/build.js` parst `README.md`, `ROADMAP.md` und `glossary/terms.md` und erzeugt `site/data.js`.
Wenn du diese Dateien bearbeitest, müssen die Parser-Muster intakt bleiben:

- Phasen-Header im Format `### Phase N: Name \`X lessons\`` oder als `<details><summary>...`-Variante.
- Lektionstabellen mit Spaltenform `| # | Lesson | Type | Lang |` (bzw. `| # | Project | Combines | Lang |` für Capstones).
- ROADMAP-Statuszeichen (`✅`, `🚧`, `⬚`) dürfen nicht durch Text ersetzt werden.

Nach Änderungen:

```bash
node site/build.js
git diff site/data.js
```

`site/data.js` sollte strukturell unverändert sein (abgesehen vom Zeitstempel).

## Arten von Beiträgen

### 1) Neue Lektion hinzufügen

Jede Lektion liegt in `phases/XX-phase-name/NN-lesson-name/`:

```text
NN-lesson-name/
├── code/           Mindestens eine lauffähige Implementierung
├── notebook/       Optionales Jupyter-Notebook
├── docs/
│   └── en.md       Pflicht: englische Dokumentation
└── outputs/        Prompts, Skills oder Agents (optional)
```

### 2) Übersetzung hinzufügen

Lege in der jeweiligen Lektion unter `docs/` eine Sprachdatei an:

```text
docs/
├── en.md    (Englisch — immer erforderlich)
├── de.md    (Deutsch)
├── zh.md    (Chinesisch)
├── ja.md    (Japanisch)
└── ...
```

Regel: Struktur wie `en.md` beibehalten, Inhalt übersetzen, Code nicht verändern.

### 3) Output hinzufügen

Wenn eine Lektion ein wiederverwendbares Artefakt liefern soll:

1. In `outputs/` der Lektion ablegen
2. Im top-level `outputs/` Index referenzieren

### 4) Bugs fixen / Lektionen verbessern

- Nicht lauffähigen Code reparieren
- Erklärungen verbessern
- Diagramme verbessern
- Veraltete Inhalte aktualisieren

## Richtlinien

- **Code muss laufen.**
- **Build from scratch first.**
- **Direkte, klare Sprache statt Fülltext.**
- **Keine unnötigen Änderungen außerhalb des Scopes.**

## PR-Prozess

1. Repository forken
2. Feature-Branch anlegen
3. Änderungen machen
4. Lokale Checks laufen lassen
5. Pull Request mit klarer Beschreibung einreichen

## Code of Conduct

Siehe [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
