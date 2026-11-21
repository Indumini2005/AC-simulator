# Smart AC Optimizer: simulation (Team Heights, Spark 2026)

Simulates one room's AC for up to 30 days and compares today's habits
(cold setpoints, AC left on in empty rooms) with the clip-on device
(22–26 °C comfort band, setback then off when empty, Dry mode when humid,
pre-cooling, remote overrides respected).

All default inputs are SAMPLE data. Results are modelled, not measured.

## Folder

| File | What it is |
|---|---|
| `dashboard.html` | Interactive dashboard. Open it in any browser, no install needed. |
| `ac_sim.py` | Same model in Python, for batch runs and charts for the deck. |
| `sample_data/occupancy_7days.csv` | Example occupancy file (`minute,count`, minute 0 = day 1 at 00:00). |
| `sample_data/weather_7days.csv` | Example weather file (`hour,t_out,rh_out`, humidity in %). |
| `requirements.txt` | Python packages for `ac_sim.py`. |

## 0. Host it on a server (simplest)

```bash
python serve.py            # http://localhost:8000, opens your browser
python serve.py --public   # also reachable from phones/laptops on the same network
```

No packages needed. Change any setting in the dashboard and the results update live.

## 1. Dashboard

Double-click `dashboard.html`. Move the sliders on the right; everything
updates live. Under "Use your own data and calibrate" you can upload your
own occupancy and weather CSVs and enter your metered kWh.

Fonts load from Google Fonts when online; offline it falls back to system fonts.

## 2. Python

```bash
pip install -r requirements.txt
python ac_sim.py                                  # 30 days, sample inputs
python ac_sim.py --ac-type inverter --seats 40 --n-ac 2
python ac_sim.py --baseline-sp "24:1"             # careful users: device savings from empty-room shut-off only
python ac_sim.py --days 7 --occupancy-csv sample_data/occupancy_7days.csv \
                 --weather-csv sample_data/weather_7days.csv
python ac_sim.py --measured-baseline-kwh 950      # calibration check
python ac_sim.py --help                           # all options
```

Writes `results/summary.csv`, `results/day_profile.png`, `results/daily_energy.png`.

## Using your real data

1. **Occupancy:** log headcount per minute (radar, door counter, or a manual
   log every 5 min filled forward) as `minute,count`.
2. **Weather:** hourly outdoor temperature and humidity as `hour,t_out,rh_out`.
3. **Calibrate:** meter the room's AC kWh for the same days. Adjust seats,
   number/size of ACs (dashboard) or `--ua`, `--cop`, `--cap-kw` (Python)
   until the simulated *baseline* is within 15% of the meter.
4. Only then quote the saving %, labelled "simulated, calibrated to [room]".

## Model in brief

- Room: air + building-mass thermal model with a moisture balance; heat from
  walls, air leakage, people (75 W + moisture each), lights, equipment, sun.
- AC: rated capacity, COP falling as outdoor/indoor gap grows, own thermostat
  with 3-min compressor protection; fixed-speed or inverter.
- Device: can only do what an IR remote can (on/off, setpoint, mode).
