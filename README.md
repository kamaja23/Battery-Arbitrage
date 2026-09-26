# Wattson

Would a Base battery pay for itself in your ERCOT zone? Wattson backtests
Base Power's home batteries against real ERCOT settlement prices, compares
dispatch strategies against a no-battery baseline and a perfect-foresight
ceiling, ranks every Texas load zone, and plans the next day from a price
forecast.

Every figure is computed from **historical** prices and labelled that way. The
next-day plan is a forecast and is shown alongside how accurate that forecast
has actually been.

## Quick start (Docker)

```bash
docker compose up -d --build
# seed the cache volume once so the demo works without the ERCOT API
docker run --rm -v wattson_cache:/dst -v "$PWD/data/cache":/src:ro alpine sh -c 'cp -n /src/*.csv /dst/'
```

Open <http://localhost:8502>. The container needs `ERCOT API Keys.txt` in the
project root only to download windows that are not cached yet.

After changing code, rebuild **and recreate** the container. A plain
`docker compose build` leaves the old container running:

```bash
docker compose up -d --build --force-recreate
```

## Running locally

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/streamlit run app.py
.venv/bin/python -m pytest -q
```

## Demo walkthrough

The page is written for someone who has never heard of ERCOT. Prices are shown
in cents per kWh (the unit on a home bill) and earnings in dollars for the
chosen battery. Industry units live under "Show the detailed numbers".

1. **How it works**: three sentences on why a battery can earn money at all.
2. **Headline**: what a Base Core in the Austin Energy area would have kept
   over the last month up to today, and that pace scaled to a year. Presets
   run from the last 7 days to the last year, or pick custom dates.
3. **A day in the life**: pick any day to see the price of power and the
   moments the battery bought and sold. It opens on the best day.
4. **What it made each day**: green and red bars, plus how much of the money
   came from a few spiky days.
5. **How smart does the battery need to be?**: no battery, a simple rule, the
   forecast planner Wattson uses, and perfect hindsight as a yardstick.
6. **What tomorrow might look like**: the forecast for the next day, the plan,
   and how accurate the forecast has been.
7. **Compare all of Texas** (sidebar): the same battery in every pricing area,
   in dollars per year.

The page is driven by the **forecast planner**. Across April–August, after
battery wear, it beat the simple rule in 32 of 40 zone-months.

## How prices are loaded

Finished days are saved in one file per calendar month and never change.
Today's prices are still being published, so they are always fetched live and
never saved; the page shows how current they are. Any date range is assembled
from those pieces, so with no internet the app still works, just without
today, and says so. A day saved before it had finished is downloaded again;
the few 15-minute prices ERCOT itself never published are accepted as they are.

## Command line

Each command reads the cache first and needs credentials only on a cache
miss.

```bash
wattson backtest                          # Austin Energy, demo window, every Base model
wattson backtest --zone LZ_WEST --start 2026-07-01 --end 2026-07-31 --battery base_core
wattson compare-zones --hubs              # rank every zone for one battery
wattson stability                         # do zone rankings hold month to month?
wattson fetch --zone LZ_CPS --start 2026-09-01 --end 2026-09-24
wattson verify-data                       # check row counts and DST handling
```

`wattson-backtest`, `wattson-fetch` and `wattson-verify-data` are shortcuts for the
matching `wattson` subcommand. From a source checkout, use
`PYTHONPATH=src python -m wattson.cli <command>`.

## Real-time vs day-ahead prices

ERCOT, the Texas grid operator, sells wholesale electricity in two markets.
The app lets you backtest against either.

- **Real-time (RTM)**: what electricity actually sold for, reset every 15
  minutes as supply and demand shift. It swings the most, including the
  occasional spike, which is where a battery earns most of its money. This is
  the default.
- **Day-ahead (DAM)**: prices agreed the day before for each hour of the next
  day, based on forecasts. Smoother and more predictable, with fewer spikes to
  profit from.

## How the numbers work

| Row | What it is |
|---|---|
| no battery (baseline) | $0. With energy-only pricing and no battery there is no spread to capture. |
| threshold | Charges in the bottom quartile of the trailing 24 h of prices, discharges in the top quartile. Uses only past prices. |
| forecast | Each morning, forecasts the day from earlier days only, optimizes a schedule against that forecast (charging for battery wear), then follows it at real prices. |
| perfect foresight | Linear program with every future price known. An unreachable ceiling. The capture ratio is each strategy's share of it. |

- **Forecast**: the median price at each time of day over the previous 7 days
  of data. It is scored against "each day repeats the one before" on the same
  intervals. Causality is tested: changing later prices never changes an
  earlier forecast or decision.
- **Revenue per kW-year** is after battery wear, annualized from the window.
  It is not a projection.
- **Not modelled**: demand charges, solar self-consumption, backup value,
  ancillary services, and Base's fleet-level grid services. These are
  usually worth more to a homeowner than energy arbitrage alone.

## Base battery specs used

| Model | Capacity | Power | Source |
|---|---|---|---|
| Base Core | 39.2 kWh | 11 kW | basepowercompany.com/specs, Base help center |
| Base Core, two units | 78.4 kWh | 22 kW | capacity published; power assumed as 2 × 11 kW |
| Base ground-mounted | 25 kWh | 11 kW | basepowercompany.com/specs, Base help center |
| Base ground-mounted, double | 50 kWh | 11 kW | basepowercompany.com/specs, Base help center |

Base does not publish round-trip efficiency, usable charge window or cycle
life. The model assumes 90% round-trip efficiency, a 10–95% charge window, and
$0.012 per kWh of throughput for wear, all typical for LFP batteries. These
live in `src/wattson/config.py`.

## What the data says (real-time prices, April–September 2026, Base Core)

- **Zone rankings are only moderately stable.** Across April–August the mean
  rank correlation between months is 0.44. West Texas (`LZ_WEST`) is the only
  zone that finished in the top three every month.
- **Austin Energy is below average for pure arbitrage.** It never ranked above
  5th of 8. After wear, the rule-based strategy lost money there in May, June
  and July and earned in April and August.
- **Earnings swing more than rankings.** Austin Energy ranged from +$9.34 to
  −$5.13 per kW-year from one month to the next.
- **The forecast beat "same as yesterday" in every month tested**, removing
  5–29% of its error. After wear, the forecast strategy did better than the
  rule-based strategy in all six windows tested for Austin Energy. It earned
  more in the volatile months and lost less in the calm ones, by skipping
  trades whose forecast spread would not cover wear.

Reproduce with `wattson stability` and `wattson backtest`.

## Layout

```
src/wattson/config.py       battery specs and presets
src/wattson/zones.py        load zone / hub names and geography
src/wattson/data/           ERCOT client, normalization, on-disk cache
src/wattson/sim/            battery physics and backtest engine
src/wattson/strategies/     threshold, forecast, perfect foresight
src/wattson/forecast.py     price forecast, accuracy scoring, next-day plan
src/wattson/metrics.py      economics, strategy and zone comparisons, rank stability
src/wattson/parallel.py     runs long backtests on several CPU cores
src/wattson/ui.py           Streamlit app
src/wattson/cli.py          command line
```
