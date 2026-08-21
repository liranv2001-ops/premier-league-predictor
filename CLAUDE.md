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
│  ├─ raw/                # downloads + premier_league_raw.db + .cache/. Git-ignored.
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
- `SQLAlchemy` over **SQLite** — two databases, deliberately separate:
  `data/raw/premier_league_raw.db` is the landing zone written by `src/data_collection`
  (faithful to the sources); `data/processed/pl.db` is the cleaned, feature-ready
  database that `src/features` builds from it
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

## Data sources

Two sources, **neither requiring an API key**. This was a deliberate choice: the free
tiers of football-data.org (no player stats, current season only) and API-Football
(100 requests/day, restricted history) cannot cover five seasons of player data.

**Matches — [football-data.co.uk](https://www.football-data.co.uk/englandm.php)**
Per-season CSVs at `mmz4281/{YYZZ}/E0.csv`, history back to the 1990s. One row per
match: results, half-time scores, shots, corners, cards, bookmaker odds.

> Two behaviours worth knowing: it answers a **missing file with `300 Multiple
> Choices`**, not 404 — `raise_for_status()` will not catch it. And it really does
> return `429` under repeated access, so honour `Retry-After`.

**Players — Understat** (`/getLeagueData/EPL/{year}`)
Per-player season totals: goals, assists, minutes, shots, key passes, xG, xA, xGChain.
Covers 2014/15 onwards, ~550 players per season, **one request per season**.

> This is an undocumented XHR endpoint, not a public API — the same one Understat's own
> `league.min.js` calls. The older trick of scraping `var playersData = JSON.parse(...)`
> from the page **no longer works**; as of August 2026 the league page is a client-side
> app with no data in its HTML. If the endpoint changes,
> `src/data_collection/understat.py` raises `UnderstatFormatError` rather than writing
> an empty table.

**Two source-specific traps the collectors already handle:**

1. **Every Understat number is a string.** Uncast, ranking by goals sorts
   lexicographically and `"29"` loses to `"6"`.
2. **A transferred player appears once**, with season totals and every club listed in
   one comma-separated field (`"Aston Villa,Manchester United"`). That list is
   **alphabetical, not chronological**, so the most recent club is not recoverable from
   it. `team_slug` is therefore `NULL` for these ~60 rows; `team_slugs` and `n_teams`
   carry the full picture. Do not assume the first or last entry is meaningful.

**Still needed:** club crests and player photos. [TheSportsDB](https://www.thesportsdb.com/)
(30 requests/minute free) is the intended source; `.env.example` already has the slot.

---

## Conventions

**Type hints — mandatory.** Every function and method gets annotated parameters and an
annotated return type, including `-> None`. Enforced by `mypy` (`disallow_untyped_defs`)
and by ruff's `ANN` rules. This is not optional and not "add later".

```python
def compute_form(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame: ...
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

Data collection:

```powershell
.venv\Scripts\python.exe -m src.data_collection.cli --seasons 6        # matches + players
.venv\Scripts\python.exe -m src.data_collection.cli --source players
.venv\Scripts\python.exe -m src.data_collection.cli --force-refresh    # ignore cache

# One request per match (~2,280, about an hour). Run it in the background; it is
# resumable, so an interrupted run picks up from the cache.
.venv\Scripts\python.exe -m src.data_collection.cli --seasons 6 --source player_matches --min-interval 2.0
```

Feature building (reads the raw DB, never the network):

```powershell
.venv\Scripts\python.exe -m src.features.cli
.venv\Scripts\python.exe -m src.features.cli --only team
```

Modelling and simulation:

```powershell
.venv\Scripts\python.exe -m src.models.cli --season 2025/26 --cutoff-matchweek 20
.venv\Scripts\python.exe -m src.models.cli --season 2025/26 --cutoff-matchweek 20 --compare-variants
.venv\Scripts\python.exe -m src.models.cli --season 2026/27 --teams-file teams.txt
```

**Use `--seasons 6`, not 5.** The earliest season is consumed as history so the second
season has real previous-season features, and is then dropped from the output.

**Caching is not an optimisation here, it is the quota strategy.** Completed seasons are
immutable and cached with no expiry; only the season in progress has a TTL (12h). A cold
run is ~12 requests, a warm run is 1. Never delete `data/raw/.cache/` casually, and do
not reach for `--force-refresh` out of habit.

---

## Features

`src/features` turns the raw DB into `data/processed/pl.db`:

| Table | Grain | Contents |
| --- | --- | --- |
| `match_features` | one row per match | `elo_pre`, rolling goals/xG/PPDA/xpts, `rest_days`, `prev_season_rank`, `home_advantage`, as `home_*` / `away_*` / `diff_*` |
| `player_season_features` | player-season | per-90 rates and shrunk rates, `is_qualified` |
| `player_match_features` | appearance | rolling last-5 form and trend vs season to date |

**Two invariants hold across all three, and both are tested:**

**No leakage.** Every feature describes state *strictly before* the row's match. The
team builder walks matches in date order and folds a result into state only after
recording that match's features; the player builder uses `shift(1)` before every
rolling window. `tests/test_features.py` proves it by rewriting the final match to an
absurd scoreline and asserting no earlier row moves. Keep that test passing — a leak
produces a model that validates beautifully and predicts nothing.

**No NaN.** By construction, not by a trailing `fillna`. Thin history falls back to an
expanding mean, then to the league average; `rest_days` is capped at 14 with
`is_rest_capped` marking it; a club not in the division last season gets
`prev_season_rank = 21` plus `was_promoted`; multi-club players get the `"multi-club"`
sentinel. Each sentinel is paired with a flag so a model can treat it as a category
instead of trusting the number.

**Per-90 rates use shrinkage.** Minutes run from 1 to 3,420, so a raw rate makes a
one-minute goalscorer the league's best striker at 90 goals/90. `*_per_90_shrunk` pulls
each rate toward the league rate, weighted by minutes. **Rank on the shrunk columns**;
the raw ones are there for interpretation, not for ordering.

---

## Models

`src/models` fits Dixon-Coles and simulates a season's remaining fixtures.

**Dixon-Coles, not plain Poisson.** Each club has an attack and a defence rating;
`lambda_home = exp(gamma + attack[home] - defence[away])`. The `tau` correction adjusts
the four lowest scorelines (0-0, 1-0, 0-1, 1-1), because independent Poisson
systematically understates draws and 1-0s — exactly where football results concentrate.
Matches are weighted by exponential time decay, and the decay rate is **selected on
held-out matches**, never hard-coded.

**Three traps this code already handles — do not undo them:**

1. **The objective must stay finite everywhere.** Returning `inf` for an infeasible
   trial point makes scipy's finite-difference gradient compute `inf - inf = nan`, and
   L-BFGS-B then stops after one iteration with the *initial values still in place* — a
   fit that silently is not a fit. Infeasible regions get a smooth penalty instead, and
   `test_fit_actually_moves_off_the_initial_point` guards it.
2. **Ranking uses `np.lexsort`, not a scaled float key.** Packing points, goal
   difference and goals scored into one float puts the tie-break term at the limit of
   float64 precision once points reach three digits.
3. **Exact ties break on a per-simulation coin flip.** Sorting them alphabetically would
   hand clubs early in the alphabet a systematic edge in every tied season.

**Promoted and returning clubs** get two related treatments, both of which exist to stop
an artefact of *which clubs we happen to have observed* leaking into the predictions:

- A club with **no** history takes the promoted-club baseline, measured from each
  newcomer's **first** season only (goals for/against relative to the league mean).
  Averaging over a club's whole tenure instead lets sides that came up and then
  established themselves — Leeds, Sunderland — drag the baseline upward, which is wrong
  for a club that is arriving. First-seasons-only puts it at attack −0.373 / defence
  −0.245, against −0.121 / −0.325 for the old estimator.
- A club that **played here before but has been away** keeps its own rating blended
  toward that baseline by `staleness_weight`, a half-life of one season away with a
  120-day grace period. Without this, having data makes a club look *worse* than being
  unknown: Ipswich carried a −0.282/−0.553 rating from their 22-point 2024/25 and ranked
  below two clubs we know nothing about. The grace period is what keeps every
  established club completely untouched — there is a test for exactly that.

Never special-case a named club here. Both mechanisms are general, and the next
relegated-then-promoted side gets handled without a code change.

**The covariates from `src/features` were tried and lost.** Classic Dixon-Coles scores
-2.9101 log-likelihood per held-out match against -2.9110 for the variant that adds
ELO, previous-season rank and promotion status. Attack and defence already encode what
ELO encodes. Classic is what ships; rerun the comparison any time with
`--compare-variants`. Note that only *season-static* covariates are usable at all here —
rolling form and rest days are undefined for a fixture that has not been played.

Output: `data/processed/simulation_{season}.json`, carrying `predicted_position` (mean),
`predicted_rank` (1-20), `title_probability` and a `position_distribution` with all 20
keys present.

---

## Awards

`src/models/awards.py` predicts each player's remaining goals and assists with two
`XGBRegressor`s using `objective="count:poisson"` — these are counts, and squared error
would happily predict negative goals while chasing the handful of 20-goal seasons.

**Point estimates cannot answer "who wins the Golden Boot".** Each prediction becomes a
Poisson mean, 10,000 seasons are simulated, and the winner is counted. A `k`-way tie
splits credit `1/k`, so probabilities sum to exactly 1 — which is also the test.

**Two traps, both already handled:**

1. **Validation splits by season, never randomly.** Each player-season appears at six
   cutoffs, so a random fold would train on matchweek 25 and test on matchweek 10 of the
   same season — the same matches on both sides.
2. **A fresh season is a different question from mid-season.** Training only on
   mid-season cutoffs and then serving matchweek 0 is a train/serve mismatch: every
   feature sits outside the range the model saw, and predictions visibly compress toward
   the mean (Haaland dropped to 3rd on 12 goals after scoring 27). `build_preseason_state`
   produces both the pre-season training rows and the pre-season prediction input, so the
   two are identical by construction. Do not reintroduce the shortcut of building a
   full-season state and overriding `matches_remaining`.

Player of the Season is a weighted score — attacking 50%, team 30%, minutes 20% — over
min-max normalised components, gated at `QUALIFYING_MINUTES`. `POTS_WEIGHTS` is asserted
to sum to 1 at import.

Output: `data/processed/predictions.json`, carrying the table, champion, all three awards
with five candidates each, the validation scores, and a machine-readable `assumptions`
array. That array is not decoration — a caveat nobody can read is one that gets
forgotten, and the dashboard should surface it.

---

## Dashboard

`frontend/` renders `predictions.json`. No backend, no API — Vite serves a static file.

**Colours are validated, not chosen by eye.** The palette comes from the `dataviz`
skill's reference instance, declared as CSS custom properties in
`frontend/src/index.css` under both `@media (prefers-color-scheme: dark)` and
`:root[data-theme="dark"]`, so the theme toggle wins in both directions.

Exactly **three** colours carry meaning: categorical slot 1 (blue) for every
probability mark, status-critical (red) for relegation, and the ink/surface scale.
Both modes pass `scripts/validate_palette.js` (worst CVD ΔE 23.8 light / 25.7 dark).

**Green is deliberately absent.** The obvious design — green for Champions League,
red for relegation — fails CVD separation at ΔE 4.1 under deuteranopia: a red/green
reader cannot tell the top of the table from the bottom. UCL is a *category*, not a
status, so it takes the series blue, and both zones carry a text label ("UCL" / "REL")
so colour is never the only signal. **Do not "fix" this by adding green back.**

**Other calls the skill forces, worth not undoing:**

- The table is a table. Twenty clubs all carrying meaning is past the point colour can
  separate them — which also gives every mark on the page its accessible table-view twin.
- Probability bars are one hue, not a ramp. Shading each bar darker-where-bigger would
  double-encode length as hue on categories that have no natural order.
- The finish range is a real 80% credible interval from `position_distribution`, on a
  shared 1–20 axis so rows are comparable. Not a decorative progress bar.
- Hero figures use proportional figures; `tabular-nums` is confined to table columns.

## Assets and licensing

**The club badges are ours, not the clubs'.** `src/data_collection/club_badges.py`
generates a monogram SVG per club in that club's real colours. This is not a stylistic
choice: Premier League crests are trademarked, Wikimedia Commons excludes them by
policy, and a survey of all 20 clubs found exactly one crest SVG under a free licence —
a 1930 Arsenal crest. Every generated file carries an XML comment saying it is not the
official crest, and so does `assets/logos/mapping.json`. **Do not swap these for real
crests and call them open-licensed.** (`thesportsdb.py` still fetches the real crests if
a deliberate fair-use decision is ever taken; that is a different question.)

**Player photos are free-licensed and carry a legal attribution obligation.**
`src/data_collection/wikimedia.py` fetches them from Wikimedia Commons. Two rules hold
it together:

- **Identify the person, then take their image — never search images by text.** Text
  search returns *Mohamed Salah (football manager)*, a different man, and misses Bukayo
  Saka entirely. Resolving the player's Wikipedia article and taking its lead image gets
  the right person. The club is then checked against the article's own introduction,
  because a mononym like "Thiago" redirects to Thiago Alcantara — who never played for
  Brentford. That guard is what finds Igor Thiago instead.
- **The licence check is an allowlist that fails closed**, and it explicitly rejects
  `CC BY-NC` and `CC BY-ND`. Those open with "CC BY" and are *not* free licences; a
  prefix match alone waved them straight through until a test caught it.

Attribution lives in two places and needs both: `assets/players/mapping.json` is the
record that survives in version control, and the footer `Credits` block is what
discharges the CC BY / CC BY-SA obligation, because attribution nobody can see is not
attribution. Both mapping files **are committed** even though the images are not.

Photos are fetched at `THUMBNAIL_WIDTH` (400px), not full size — the originals are press
photographs and the eleven candidates came to 16.5 MB before, against 0.92 MB now, for
images rendered at 72px.

**Every image has a fallback that always renders.** Badges and photos are fetched
best-effort and git-ignored, so the page must look right with none of them present —
and it has to, because Nottingham Forest genuinely has no badge in TheSportsDB (a
netball club shadows it in search) and three shortlisted players have no usable photo.
`TeamBadge` and `PlayerPhoto` swap to a monogram on the image's `error` event. Note that
Vite's dev server answers a missing file with **200 and an HTML body**, not 404, so the
fallback must key off decode failure rather than status.

Rebuild and publish:

```powershell
.venv\Scripts\python.exe -m src.models.cli --season 2026/27 `
  --teams-file data/seasons/2026-27.txt --awards --publish
npm run dev --prefix frontend
```

`--publish` copies the payload into `frontend/public/` and fetches any missing images.
It is a separate step on purpose: the dashboard should never quietly serve a stale
payload that someone forgot to copy across.

---

## Backtest — read this before trusting a prediction

`src/models/backtest.py` trains on every season before X and predicts X **from matchweek
0**, then scores it against what actually happened. Run it with
`python -m src.models.cli --backtest`.

**The honest result, over 2022/23–2025/26:**

| | model | carry-forward baseline |
| --- | --- | --- |
| Mean position error | **3.60 places** | **3.58 places** |
| Champion identified | 2 of 4 | — |
| 80% interval coverage | 62% | (target 80%) |

**The model does not beat "just repeat last season's table."** It won on 1 season of 4.
It named the champion twice — both times Manchester City, and it predicted Manchester
City in *all four* seasons, so the hits came from a standing prior rather than from
reading each season. It gave the actual 2024/25 champion, Liverpool, 11.2%.

**It is also overconfident**: 62% of clubs finished inside their own 80% band, when 80%
would be calibrated. The bands are too narrow, so quoted probabilities are firmer than
the evidence supports.

None of this makes the mid-season numbers wrong — cut at matchweek 20, the model gives
Arsenal 74.6% and Arsenal won. Half a season of standings is most of the answer.
Preseason, with only prior years to go on, it is roughly as good as a much simpler rule.
**Say so when reporting predictions rather than quoting the title probability alone.**

The obvious things to try: a stronger prior on promoted and newly-strong clubs, squad
turnover as a feature (the model has no idea who was bought or sold), and widening the
simulated variance so the intervals are honest.

**The leak this harness had to avoid.** `run_season_simulation` selects the time-decay
hyperparameter against `actual_remaining`, which at a matchweek-0 cutoff is the whole
season being predicted. Unreachable in production, fatal in a backtest. `backtest.py`
holds out the last *training* season instead, and
`tests/test_backtest.py::test_decay_selection_never_sees_the_predicted_season` pins it.
**Do not route a backtest through `run_season_simulation`.**

---

## Current state

**Done:** scaffold; `src/data_collection`; `src/features`; `src/models` (Dixon-Coles +
Monte Carlo). Seasons 2021/22–2025/26 are emitted, with 2020/21 collected as history.

**The 2025/26 back-test is the reference result.** Cut at matchweek 20 and simulated,
the model gives Arsenal a 74.6% title probability and 85.05 expected points; Arsenal
actually won with 85. All three relegated clubs land in 18-20. Mean absolute rank error
is 2.2. If a change moves those numbers materially, something broke.

**2026/27 is predicted.** The club list lives at `data/seasons/2026-27.txt` — a fact the
pipeline cannot derive, so it is committed next to the code that reads it. Promoted:
Coventry, Hull City, Ipswich. Relegated: West Ham, Wolves, Burnley (which our own data
independently derives as the bottom three, and it agrees).

Since zero matches have been played, the full 380-fixture double round-robin is
simulated and **fixture order is irrelevant** to the final table — no schedule needed.
Once matches start, re-run with `--cutoff-matchweek` to fold in real results.

**Award models are done and validated.** Back-tested on 2025/26 cut at matchweek 20:
Haaland ranked 1st for the Golden Boot at 95.4% with 26.9 predicted goals — he actually
scored 27. Bruno Fernandes ranked 1st for assists at 37.6%, and did top the assist chart.
Both regressors beat the "current rate simply continues" baseline. **If a change moves
those numbers materially, something broke.**

One honest limitation: the model predicted Bruno on 10 assists and he finished on 21. He
added 14 after matchweek 20 — the highest second-half assist run in six seasons, against
a 75th percentile of 1. A Poisson model trained on that distribution cannot produce it.
Rankings are reliable; extreme magnitudes are not.

**The dashboard is live and wired to the real payload.** 19/20 club badges and 8/11
player photos are fetched; the rest render as monograms by design.

The original brief is complete end to end: collection → features → team model → award
models → dashboard.

**Known gaps, none blocking:**

- Nottingham Forest has no badge in TheSportsDB, and three shortlisted players
  (Thiago, Mohamed Salah, Morgan Rogers) have no photo that could be confirmed as the
  right person. All five fall back to monograms rather than showing the wrong face.
- The dashboard has no automated UI tests. It is verified through the browser tools and
  a Python test asserting `predictions.json` carries every field the page reads.

## Environment note

This machine runs low on free memory (~1.4 GB observed). OpenBLAS intermittently fails
with "Memory allocation still failed after 10 retries". If a run dies that way, set
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` — it is a threading/allocation issue,
not a bug in the model code.
