# FIFA World Cup 2026 Fantasy Team Optimizer

Builds the best possible 15-man fantasy squad for the FIFA World Cup 2026 under
the game's budget, position, country, and formation rules — and recommends
transfers and boosters as the tournament progresses.

## Core idea

The problem is **two separate problems**, kept separate on purpose:

1. **Projection (statistics / ML):** estimate each player's expected fantasy
   points (EP) for the next round from historical stats.
2. **Selection (exact optimization, not ML):** pick the 15 players that maximize
   EP under all constraints. This is an **Integer Linear Program** solved
   *provably optimally* in seconds with PuLP/CBC. ML is the wrong tool here.

```
scrape FBref ─┐
              ├─► entity match ─► project EP ─► ILP optimizer ─► squad / XI / captain
your prices ──┘                                    ▲
                                          rules: budget / 2-5-5-3 /
                                          country caps / formations
matchday results ─► re-scrape ─► re-project ─► transfer + booster planner
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick start (no network needed)

Runs the entire pipeline on synthetic data:

```bash
python -m fantasy demo
```

Or launch the dashboard:

```bash
streamlit run app/streamlit_app.py
```

## Using real data

1. **Provide your price list** (`players.csv`) with columns `Name, Position,
   Country, Price`. Country may be a FIFA 3-letter code (ENG, FRA, ...) and price
   may be formatted like `$10.5m`. `Position` is optional — if omitted, positions
   are taken from the scraped FBref data via the entity match. Prices are never
   scraped.
2. **Scrape history** from FBref (needs internet). Uses **league-level scraping**
   (one request per stat category per competition — a handful of throttled
   requests covering thousands of players) for a **single most-recent season
   (2025/26)**, summed per player for an "all competitions" view:

   ```bash
   python -m fantasy scrape                       # big-5 + Euro 2024
   python -m fantasy scrape --season 2526 --leagues "Big 5 European Leagues Combined"
   ```

   ### Data coverage caveat
   `soccerdata`'s FBref integration natively supports only the big-5 European
   leagues (combined) plus past World Cups / Euros. It does **not** expose
   domestic cups, the Champions/Europa League, leagues outside the big 5 (MLS,
   Saudi, Eredivisie, Liga MX, ...), or this-season WC qualifiers / Nations
   League / friendlies. So a big-5-only scrape matches ~50% of a global World Cup
   pool; the rest get EP 0 until covered. To widen coverage, pass more leagues to
   `build_history(club_leagues=[...])` / `--leagues` (any soccerdata-supported
   FBref league) or extend soccerdata's `league_dict.json`. FBref is behind a
   Cloudflare challenge — run from a residential IP if you hit a 403.

3. **Optimize** your squad for a stage:

   ```bash
   python -m fantasy optimize --prices players.csv --stage group
   ```

4. **Live updates** — after each round, re-scrape and get transfer + booster
   recommendations (squad file = current player_ids, one per line):

   ```bash
   python -m fantasy live --prices players.csv --squad my_squad.txt --stage r16 --free 4
   ```

## Project layout

| Path | Purpose |
| --- | --- |
| `src/fantasy/rules.py` | All game constants (budget, composition, caps, formations, transfers) |
| `src/fantasy/scoring/` | Scoring engine: raw match stats → fantasy points (single source of truth) |
| `src/fantasy/optimize/` | ILP squad optimizer + transfer/booster planner |
| `src/fantasy/ingest/` | FBref scraper, price loader, entity matcher, storage |
| `src/fantasy/projection/` | Minutes model, EP model, Qatar-2022 backtest |
| `src/fantasy/pipeline/` | End-to-end run + live recompute loop |
| `app/streamlit_app.py` | Dashboard |
| `tests/` | Unit + integration tests |

## How the optimizer encodes the rules

- Budget `$100m` (group) / `$105m` (knockouts); exactly 2 GK, 5 DEF, 5 MID, 3 FWD.
- Per-country cap scales by stage (3 → 8).
- Starting XI = 1 GK + outfield lines bounded DEF∈[3,5], MID∈[3,5], FWD∈[1,3],
  which reproduces *exactly* the 7 legal formations — no extra variables needed.
- Captain = highest-EP starter (doubled in the objective).
- Transfers beyond the free allowance cost −3 points each (modelled in the ILP).

## Modelling notes / assumptions

- **Clean sheets** require 60+ minutes (applied consistently to GK/DEF/MID).
- **Threshold rewards** (every 3 saves, every 2 shots on target, etc.) are scored
  exactly per the rules, and projected in expectation as `expected_count /
  threshold`.
- **Scope:** one most-recent season, summed across all scraped competitions;
  international appearances are weighted higher than club (more representative of
  the World Cup) via `ProjectionConfig.world_cup_weight`.
- **Minutes model** is a transparent proxy (minutes-per-game → participation
  probability); it's the obvious place to plug in a learned model later.
- **Clean-sheet / concede** use a Poisson model on team goals-against; supply
  `team_ga_override` per country to calibrate.
- **Scouting bonus** (<5% ownership) is omitted until ownership data exists.
- The projection is intentionally interpretable; its component expectations are
  exactly the features for a LightGBM/XGBoost upgrade, localized to
  `projection/model.py`.

## Tests

```bash
pytest
```
