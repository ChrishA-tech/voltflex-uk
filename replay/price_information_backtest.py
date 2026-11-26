"""Compare EV charging policies under different tariff information on archived prices."""

import argparse
from datetime import timedelta
import json
from pathlib import Path
from statistics import mean

from replay import dispatch
from tariff_archive import DAYS, REGIONS, ROOT, START


def prices(archive, code, day):
    start = START + timedelta(days=day)
    result = [
        float(archive[code][(start + timedelta(minutes=30 * slot)).strftime("%Y-%m-%dT%H:%M:%SZ")]) / 100
        for slot in range(16)
    ]
    if not all(-5 <= value <= 5 for value in result):
        raise ValueError("Invalid tariff in charging window")
    return result


def evaluate(archive):
    cases = []
    for code, region in REGIONS.items():
        history = [prices(archive, code, day) for day in range(DAYS)]
        for day in range(14, DAYS):
            observed = history[day]
            expected = [mean(slot) for slot in zip(*history[day - 14:day], strict=True)]
            earliest, historical = dispatch(observed, expected)
            _, published = dispatch(observed, observed)
            cases.append({
                "date": (START + timedelta(days=day)).date().isoformat(),
                "region": region,
                "earliest": earliest,
                "historical": historical,
                "published": published,
            })
    if any(row["published"] - min(row["historical"], row["earliest"]) > 1e-8 for row in cases):
        raise ValueError("Known-tariff optimum was exceeded by another policy")
    worst = max(cases, key=lambda row: row["historical"] - row["earliest"])
    return {
        "source": "Octopus Agile historical tariffs, simulated 6 kWh EV sessions",
        "metric": "GBP cost per session; lower is better",
        "holdout": ["2025-10-20", "2025-11-02"],
        "cases": len(cases),
        "mean_cost_gbp": {key: round(mean(row[key] for row in cases), 6)
                           for key in ("earliest", "historical", "published")},
        "historical_forecast_wins": sum(row["historical"] < row["earliest"] - 1e-8 for row in cases),
        "historical_forecast_losses": sum(row["historical"] > row["earliest"] + 1e-8 for row in cases),
        "largest_historical_forecast_disadvantage": {
            "date": worst["date"], "region": worst["region"],
            "extra_cost_gbp": round(worst["historical"] - worst["earliest"], 6),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = json.loads((ROOT / "archive/octopus-october-2025.json").read_text(encoding="utf-8"))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evaluate(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, KeyError, ValueError, TypeError) as error:
        parser.exit(2, f"Tariff information backtest failed: {error}\n")


if __name__ == "__main__":
    main()
