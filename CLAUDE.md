# CLAUDE.md — premier-league-predictor

## Project goal

An ML system that predicts the **final Premier League table** for a season and the
season's individual awards, presented in a styled dashboard.

Predicted outputs:

| Output | Type |
| --- | --- |
| Final table, positions 1–20 | ranking (per-team points/rank) |
| Champion | classification (derived from rank 1) |
| Top scorer (Golden Boot) | ranking over players |
| Top assists | ranking over players |
| Player of the Season | ranking over players |

The dashboard shows club **logos** and **player photos** alongside the predictions.

---

## Directory layout

```
premier-league-predictor/
├─ CLAUDE.md              # this file
├─ requirements.txt       # Python deps (runtime + dev in one file)
├─ pyproject.toml         # pytest / ruff / mypy config only — project is NOT packaged
├─ .venv/                 # local virtualenv (git-ignored)
├─ data/
│  ├─ raw/                # downloaded CSVs, untouched. Git-ignored.
│  └─ processed/          # pl.db (SQLite) + derived tables. Git-ignored.
├─ src/
│  ├─ data_collection/    # download + parse external sources -> data/raw -> SQLite
│  ├─ features/           # feature engineering: SQLite/DataFrame -> model matrices
│  └─ models/             # training, evaluation, prediction, JSON export
├─ frontend/              # Vite + React + TypeScript + Tailwind dashboard
├─ assets/
│  ├─ logos/              # club crests. Git-ignored (binary, re-fetchable).
│  └─ players/            # player photos. Git-ignored.
└─ tests/                 # pytest suite
```

**Layer boundaries — keep these strict:**

- `src/data_collection` is the *only* place that touches the network or writes to
  `data/raw` and `data/processed`. Nothing downstream re-downloads.
- `src/features` reads from SQLite/parquet and returns DataFrames or arrays. It must
  never fit a model and never hit the network.
- `src/models` consumes feature matrices, trains/predicts, and writes the export JSON.
  It must never do feature engineering inline — if a transformation is needed, it
  belongs in `src/features`.

`data/` and `assets/` keep `.gitkeep` files so the structure survives in git while
their contents stay ignored.

---

## Stack

**Python 3.14** (venv at `.venv/`)

- `pandas` 3.x, `numpy` — data handling
- `scikit-learn`, `xgboost` — modelling
- `SQLAlchemy` over **SQLite** — storage at `data/processed/pl.db`
- `requests`, `beautifulsoup4`, `lxml` — data collection
- `joblib` — model persistence
- `pytest`, `ruff`, `mypy` — dev

> **pandas 3.x note:** copy-on-write is the default and the string dtype changed from
> `object` to a dedicated string type. Do not copy pandas 2.x idioms (chained
> assignment, `inplace=True`) from older tutorials.

**Frontend**

- Vite + React + TypeScript
- **Tailwind CSS v4** via the `@tailwindcss/vite` plugin. There is deliberately
  **no `tailwind.config.js` and no `postcss.config.js`** — v4 is configured through
  `@import "tailwindcss";` in `src/index.css` and, when needed, a `@theme` block in
  that same file. Don't add a v3-style config; it will be ignored.

---

## Data source

**Primary: [football-data.co.uk](https://www.football-data.co.uk/englandm.php)** —
free per-season CSVs (`E0.csv`), no API key, history back to the 1990s. Each row is
one match: teams, full-time/half-time result, shots, shots on target, corners, cards,
and bookmaker odds.

**Known gap — this is the main open question for the next phase:**
football-data.co.uk is *match-level only*. It contains **no player statistics**, so it
cannot produce the top scorer, top assists, or player-of-the-season predictions on its
own. Those need a supplementary source (FBref or Understat for per-player goals,
assists and xG). That source has **not been chosen yet** — decide before building
`src/features` for the player-level targets.

The table/champion predictions can be built end-to-end from football-data.co.uk alone,
so start there.

---

## Conventions

**Type hints — mandatory.** Every function and method gets annotated parameters and an
annotated return type, including `-> None`. Enforced by `mypy` (`disallow_untyped_defs`)
and by ruff's `ANN` rules. This is not optional and not "add later".

```python
def compute_form(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    ...
```

**Tests for `src/models`.** Anything added under `src/models` must have a matching test
in `tests/`. Feature and collection code is tested where practical; model code is
tested always — a silently wrong model is the failure mode this project cannot detect
by eye.

**Docstrings.** Google style, on every public function. One-line summary minimum.

**Reproducibility.** A single `RANDOM_SEED = 42` is the source of truth for every
`random_state` / seed in the project. Two runs on the same data must produce identical
predictions.

**Never commit:** raw or processed data, `*.db`, trained models (`*.pkl`, `*.joblib`),
logos, or player photos. All are git-ignored — the pipeline re-creates them.

**Asset naming.** Kebab-case slugs, so Python and React agree without a lookup table:

```
assets/logos/manchester-united.png
assets/players/erling-haaland.jpg
```

The same `slug` field appears in the export JSON. Team names from
football-data.co.uk are inconsistent (`Man United`, `Nott'm Forest`) — normalise
to slugs once, in `src/data_collection`, and use slugs everywhere downstream.

**Line length 100.** Run `ruff format` and `ruff check` before committing.

---

## Data contract (Python → React)

There is **no API server**. The pipeline writes a single static file:

```
frontend/public/predictions.json
```

React fetches it with `fetch('/predictions.json')`. This keeps the project to one
runtime (Vite) during development and makes the dashboard deployable as static files.
If a server ever becomes necessary, that is a deliberate decision to revisit — do not
add FastAPI casually.

Schema:

```jsonc
{
  "generated_at": "2026-08-20T12:00:00Z",
  "season": "2026-27",
  "model_version": "xgb-v1",
  "table": [
    {
      "position": 1,
      "team": "Manchester City",
      "slug": "manchester-city",
      "predicted_points": 88.4,
      "played": 38,
      "goal_difference": 52,
      "confidence": 0.71
    }
    // ... 20 entries, ordered by position
  ],
  "champion":            { "team": "Manchester City", "slug": "manchester-city", "probability": 0.42 },
  "top_scorer":          { "player": "Erling Haaland", "slug": "erling-haaland", "team_slug": "manchester-city", "predicted_goals": 27.3 },
  "top_assists":         { "player": "...", "slug": "...", "team_slug": "...", "predicted_assists": 14.1 },
  "player_of_the_season":{ "player": "...", "slug": "...", "team_slug": "...", "score": 0.91 }
}
```

Images resolve from the slug: `/logos/{slug}.png`, `/players/{slug}.jpg`. Assets are
copied (or symlinked) from `assets/` into `frontend/public/` as a build step — the
React app never reads `assets/` directly.

---

## Commands

Python (from the project root):

```powershell
.venv\Scripts\Activate.ps1          # activate the venv
.venv\Scripts\python.exe -m pytest  # run tests
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format .
.venv\Scripts\python.exe -m mypy
```

Frontend:

```powershell
npm run dev --prefix frontend       # dev server on http://localhost:5173
npm run build --prefix frontend
```

The dev server is also registered in `.claude/launch.json` as **`pl-dashboard`**.

---

## Current state

Scaffold only. `frontend/src/App.tsx` is a placeholder shell with a static 20-row table
and four award cards — it does **not** fetch `predictions.json` yet. There is no
collection, feature, or model code; `src/` contains empty packages. `tests/test_smoke.py`
only verifies that the environment imports.

Next steps, in order:

1. Choose the player-statistics source (see "Data source" above).
2. `src/data_collection`: download football-data.co.uk CSVs → normalise team slugs →
   load into `data/processed/pl.db`.
3. `src/features`: per-team rolling form, home/away splits, goal difference, xG proxies.
4. `src/models`: train, evaluate against held-out seasons, export `predictions.json`.
5. Wire the dashboard to the real JSON and fetch logos/photos.
