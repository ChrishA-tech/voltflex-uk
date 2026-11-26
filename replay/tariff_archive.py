"""Benchmark charging against archived published Octopus Agile half-hour tariffs."""
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlopen

from replay import dispatch, report

START = datetime(2025, 10, 6, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parent
REGIONS = {"A": "Eastern", "C": "London", "G": "North West"}
DAYS = 28
PRODUCT = "AGILE-24-10-01"


def download(path):
    stop = START + timedelta(days=DAYS)
    payload = {}
    for code in REGIONS:
        url = (f"https://api.octopus.energy/v1/products/{PRODUCT}/electricity-tariffs/"
               f"E-1R-{PRODUCT}-{code}/standard-unit-rates/?page_size=1500"
               f"&period_from={START.isoformat().replace('+00:00', 'Z')}"
               f"&period_to={stop.isoformat().replace('+00:00', 'Z')}")
        with urlopen(url, timeout=25) as response:
            data = json.load(response)
        if data["next"] is not None:
            raise ValueError("Tariff response exceeded one page")
        payload[code] = {row["valid_from"]: row["value_inc_vat"] for row in data["results"]}
    list(replay_archive(payload))  # Validate the complete window before saving.
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def replay_archive(payload):
    if payload.keys() != REGIONS.keys():
        raise ValueError("Expected exactly three tariff regions")
    for day in range(DAYS):
        start = START + timedelta(days=day)
        for code, name in REGIONS.items():
            key = lambda slot: (start + timedelta(minutes=30 * slot)).strftime("%Y-%m-%dT%H:%M:%SZ")
            daily = [float(payload[code][key(slot)]) / 100 for slot in range(48)]
            if not all(-5 <= price <= 5 for price in daily):
                raise ValueError("Tariff outside expected replay units of pounds per kWh")
            prices = daily[:16]
            first, second = dispatch(prices, prices, goal=6, limit=2)
            scenario = "Negative prices" if min(prices) < 0 else "Non-negative prices"
            yield day + 1, name, scenario, first, second, 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Fetch the fixed archive from Octopus")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = ROOT / "archive/octopus-october-2025.json"
    try:
        if args.refresh:
            download(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = report(replay_archive(payload), "Observed Octopus Agile prices, 6 October–2 November 2025; simulated EV sessions")
        result["config"] = dict(result["config"], slug="voltflex-uk", name="VoltFlex UK",
                               subtitle="Compare charging policies on 28 days of archived Octopus tariffs.",
                               note="The first eight hours of each UTC day use published half-hour prices. EV sessions and the 6 kWh charge target are simulated. This replay assumes the full day-ahead tariff is available at midnight.")
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, KeyError, ValueError, TypeError) as error:
        parser.exit(2, f"Archive replay failed: {error}\n")


if __name__ == "__main__":
    main()
