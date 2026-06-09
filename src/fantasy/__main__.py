"""Command-line interface for the FIFA World Cup 2026 Fantasy optimizer.

Examples:
  python -m fantasy demo
  python -m fantasy scrape                      # build history from FBref (needs internet)
  python -m fantasy optimize --prices prices.csv --stage group
  python -m fantasy live --prices prices.csv --squad ids.txt --stage r16 --free 4
"""

from __future__ import annotations

import argparse
import sys

from fantasy.rules import Stage


def _load_history(path: str | None):
    from fantasy.ingest.storage import load_table
    if path:
        import pandas as pd
        return pd.read_parquet(path)
    return load_table("player_history")


def cmd_demo(_args):
    from fantasy.pipeline.run import run_demo
    run_demo(stage=Stage(_args.stage))


def cmd_scrape(args):
    from fantasy.ingest.fbref import DEFAULT_CLUB_LEAGUE, build_history
    leagues = [s.strip() for s in args.leagues.split(";")] if args.leagues else [DEFAULT_CLUB_LEAGUE]
    df = build_history(season=args.season, club_leagues=leagues, save=True)
    print(f"Scraped {len(df)} rows across {len(leagues)} league(s) -> data/processed/player_history.parquet")


def cmd_optimize(args):
    from fantasy.pipeline.run import run_pipeline
    history = _load_history(args.history)
    solution, report = run_pipeline(args.prices, history, stage=Stage(args.stage))
    print(f"Entity match: {report['matched']}/{report['total']} ({report['match_rate']:.0%})")
    print(solution.summary())


def cmd_live(args):
    from fantasy.ingest.prices import load_price_list
    from fantasy.pipeline.live import describe_update, recommend_update

    history = _load_history(args.history)
    price_df = load_price_list(args.prices)
    with open(args.squad) as f:
        squad_ids = {line.strip() for line in f if line.strip()}
    free = args.free if args.free >= 0 else None
    update = recommend_update(squad_ids, price_df, history, Stage(args.stage), free)
    print(describe_update(update))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fantasy")
    sub = parser.add_subparsers(dest="command", required=True)
    stages = [s.value for s in Stage]

    p = sub.add_parser("demo", help="Run end-to-end on synthetic seed data")
    p.add_argument("--stage", choices=stages, default="group")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("scrape", help="Scrape history from FBref (needs internet)")
    p.add_argument("--season", default="2526", help="FBref season code, e.g. 2526")
    p.add_argument("--leagues", default=None, help="';'-separated soccerdata league names")
    p.set_defaults(func=cmd_scrape)

    p = sub.add_parser("optimize", help="Optimal squad from a price list")
    p.add_argument("--prices", required=True)
    p.add_argument("--history", default=None, help="Parquet history path (defaults to processed table)")
    p.add_argument("--stage", choices=stages, default="group")
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser("live", help="Recommend transfers + booster for the next round")
    p.add_argument("--prices", required=True)
    p.add_argument("--squad", required=True, help="Text file of current squad player_ids, one per line")
    p.add_argument("--history", default=None)
    p.add_argument("--stage", choices=stages, default="group")
    p.add_argument("--free", type=int, default=-1, help="Free transfers (-1 = unlimited)")
    p.set_defaults(func=cmd_live)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
