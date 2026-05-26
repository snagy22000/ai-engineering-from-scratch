# AI Engineering from Scratch (Deutsch)

> Deutsche Einstiegsseite für das Curriculum. Das vollständige Original bleibt in `README.md`.

## Kurzüberblick

- 20 Phasen
- 435 Lektionen
- Sprachen im Curriculum-Code: Python, TypeScript, Rust, Julia
- Jede Lektion erzeugt ein nutzbares Artefakt (Prompt, Skill, Agent oder MCP-Server)

## So startest du

### Option A – direkt lesen

- Curriculum-Webseite: https://aiengineeringfromscratch.com
- Englisches Inhaltsverzeichnis: [README.md](README.md#contents)
- Fortschrittsübersicht: [ROADMAP.md](ROADMAP.md)

### Option B – lokal ausführen

```bash
git clone https://github.com/rohitg00/ai-engineering-from-scratch.git
cd ai-engineering-from-scratch
python phases/01-math-foundations/01-linear-algebra-intuition/code/vectors.py
```

### Option C – Level bestimmen

```bash
/find-your-level
```

## Deutschsprachige Beiträge

Dieses Repository unterstützt Übersetzungen pro Lektion:

- Jede Lektion enthält `docs/en.md` (Pflicht)
- Deutsche Übersetzung wird als `docs/de.md` ergänzt
- Struktur und Abschnittsreihenfolge müssen mit `en.md` übereinstimmen

Details zum Mitwirken stehen in [CONTRIBUTING.de.md](CONTRIBUTING.de.md).

## Vorschlag für schrittweise Vollübersetzung

1. Phase auswählen (`phases/XX-...`)
2. Pro Lektion `docs/de.md` ergänzen
3. Inhalte übersetzen, Code-Blöcke unverändert lassen
4. Mit den Repository-Checks prüfen

```bash
python3 scripts/audit_lessons.py
python3 scripts/build_catalog.py
python3 scripts/check_readme_counts.py
```

## Hinweise

- Das englische Original (`README.md`, `ROADMAP.md`, Lektionen unter `docs/en.md`) bleibt die Referenzquelle.
- Diese Datei bietet einen deutschsprachigen Einstieg und Übersetzungsworkflow.
