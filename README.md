# Premier League Predictor

Predicts the final Premier League table, the champion, and the individual awards for a
whole season — then shows it in a static React dashboard. A Dixon-Coles scoreline model
produces expected goals for every fixture; 10,000 Monte Carlo seasons turn those into a
distribution over final positions rather than a single guess.

The dashboard shows the full predicted table with each club's title probability and its
80% finish range, a hero card for the predicted champion, and three award cards with
player photos. Run it locally with `npm run dev --prefix frontend` — see
[Quickstart](#quickstart).

---

## Current prediction — 2026/27

| # | Club | Title chance | Expected points |
|---|------|--------------|-----------------|
| 1 | Manchester City | 60.5% | 81.4 |
| 2 | Arsenal | 25.5% | 76.1 |
| 3 | Liverpool | 12.6% | 72.5 |
| 4 | Chelsea | 0.7% | 60.5 |
| … | | | |
| 18 | Hull | — | 32.8 |
| 19 | Coventry | — | 32.8 |
| 20 | Ipswich | — | 28.4 |

**Top scorer** Erling Haaland (15.7 goals, 63.0%) · **Top assists** Bruno Fernandes
(7.1) · **Player of the season** Erling Haaland

Read the next section before taking any of that too seriously.

---

## How good is it, really?

Backtested by training on every season before X and predicting X **from matchweek 0**,
across 2022/23–2025/26:

| | Model | Baseline |
|---|---|---|
| Mean final-position error | **3.60 places** | **3.58 places** |
| Champion identified | 2 of 4 | — |
| 80% interval coverage | 62% | 80% = calibrated |

The baseline is *"just repeat last season's table"*, with promoted clubs slotted into the
relegated clubs' places.

**The model does not beat that baseline.** It won on one season out of four. It named the
champion twice — both Manchester City — but it predicted Manchester City in *all four*
seasons, so those hits come from a standing prior rather than from reading each season.
It gave Liverpool, the actual 2024/25 champion, 11.2%.

It is also **overconfident**: only 62% of clubs finished inside their own 80% credible
interval, so the quoted probabilities are firmer than the evidence supports.

Mid-season is a different story. Cut 2025/26 at matchweek 20 and the model gives Arsenal
74.6% — and Arsenal won. Half a season of standings is most of the answer. Preseason,
with only prior years to go on, it is roughly as good as a much simpler rule.

Known ways forward: squad turnover as a feature (the model has no idea who was bought or
sold), a stronger prior for newly-strong clubs, and widening the simulated variance so
the intervals are honest.

```bash
python -m src.models.cli --backtest    # reproduces the table above
```

---

## Quickstart

Requires Python 3.14+ and Node 20+.

```bash
git clone <repo-url>
cd premier-league-predictor

py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt   # PowerShell
# source .venv/bin/activate && pip install -r requirements.txt  # bash
```

Then run the pipeline in order:

```bash
# 1. Collect match results and player statistics (~14 requests, cached afterwards)
python -m src.data_collection.cli --seasons 6

# 2. Per-match player data. One request per match (~2,280) - slow, and cached forever.
#    Only needed for the individual awards; skip it for the table alone.
python -m src.data_collection.cli --seasons 6 --source player_matches --min-interval 2.0

# 3. Build features
python -m src.features.cli

# 4. Fit the models, predict, and publish to the dashboard
python -m src.models.cli --season 2026/27 --teams-file data/seasons/2026-27.txt \
    --awards --squad-season 2025/26 --publish

# 5. Generate club badges and fetch free-licensed player photos
python -m src.data_collection.cli --source assets

# 6. Run the dashboard
npm install --prefix frontend
npm run dev --prefix frontend        # http://localhost:5173
```

No API keys are needed. `.env.example` documents optional ones for future sources.

If a run dies with `OpenBLAS error: Memory allocation still failed`, set
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` — it is a threading issue on
low-memory machines, not a bug in the model.

---

## Methodology

### Dixon-Coles

Every club gets an attack and a defence rating. Expected goals for a fixture are

```
λ_home = exp(γ + attack[home] − defence[away])
λ_away = exp(    attack[away] − defence[home])
```

fitted by maximum likelihood over historical matches, where `γ` is home advantage.

Plain Poisson gets football wrong in a specific way: it understates draws and 1–0s, which
is exactly where results concentrate. Dixon-Coles applies a correction `τ` to the four
lowest scorelines (0-0, 1-0, 0-1, 1-1) to fix it.

Matches are weighted by exponential time decay so recent seasons count for more. **The
decay rate is selected on held-out matches**, not chosen by hand.

Newly promoted clubs have no history, so they inherit an average promoted-club rating
measured from what promoted clubs actually do in their *first* season. A club returning
after relegation keeps its own rating blended toward that baseline in proportion to how
stale it is.

### Monte Carlo simulation

Each remaining fixture is sampled 10,000 times from its **joint** score matrix — not two
independent Poisson draws, which would throw away the correction the model just fitted.
Points accumulate on top of whatever has already been earned, tables are ranked with the
real Premier League tie-breaks (points, goal difference, goals scored), and the spread of
finishing positions across those 10,000 seasons is the output.

That is what makes the "80% finish range" in the dashboard real uncertainty rather than
decoration.

### Individual awards

Two XGBoost regressors with a `count:poisson` objective predict each player's remaining
goals and assists — counts, so squared error would happily predict negative goals. Those
predictions become Poisson means, and simulating them 10,000 times converts a point
estimate into "probability this player finishes top", which is the question actually
being asked.

Player of the Season is a weighted score: attacking contribution 50%, team performance
30%, minutes 20%.

---

## Layout

```
src/data_collection/   fetching and parsing; the only layer that touches the network
src/features/          ELO, rolling form, xG, per-90 rates — all strictly pre-match
src/models/            Dixon-Coles, Monte Carlo, award models, backtest harness
frontend/              Vite + React + Tailwind dashboard, reads a static JSON
tests/                 290 tests, none of which hit the network
data/seasons/          club lists the pipeline cannot derive
```

`CLAUDE.md` carries the conventions, the traps found along the way, and the reasoning
behind decisions that look arbitrary from the outside.

---

## Data sources and licensing

| Source | Used for | Terms |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/englandm.php) | match results | free, no key |
| [Understat](https://understat.com/) | player stats, xG | undocumented endpoint, used politely with caching and rate limiting |
| [Wikimedia Commons](https://commons.wikimedia.org/) | player photos | CC BY-SA 4.0 / CC BY / CC0 — **credited in the dashboard footer** |

Player photographs require attribution under their licences; the photographer, licence
and source URL for every image are recorded in `assets/players/mapping.json` and shown in
the dashboard.

**The club badges are generated by this project and are not official crests.** Premier
League crests are trademarked and are not available under an open licence — a survey of
all 20 clubs on Wikimedia Commons found exactly one, a 1930 Arsenal crest. Rather than
mislabel trademarked artwork as open, the project draws its own monograms in each club's
colours.

---

## Automation

`.github/workflows/weekly-predictions.yml` re-runs collection, features and the season
simulation every Monday and commits an updated `data/processed/predictions.json` when it
changes. It deliberately skips the per-match player collection — 2,280 requests weekly to
a free endpoint would be abusive for data that barely moves — so the awards are refreshed
locally rather than in CI.

`.github/workflows/tests.yml` runs pytest, ruff and mypy on every push.

---

## Licence

Code is MIT. Data and images belong to their respective sources under the terms above.
