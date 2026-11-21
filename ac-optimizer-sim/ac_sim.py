#!/usr/bin/env python3
"""
Smart AC Optimizer: room energy simulation
Team Heights, Spark Startup Challenge 2026

Compares one month of a room's AC energy use in two scenarios:
  BASELINE  - how rooms run today: AC switched on in the morning at a cold
              setpoint (often 18-20 C) and left on all day, occupied or not.
  SMART     - the clip-on device: off after 10 min empty, comfort band
              22-26 C, setpoint stepping, Dry mode when humid, respects
              remote overrides for 45 min.

Model (1-minute steps):
  * Room = 2-node thermal model (air + building mass) plus a moisture balance.
  * Heat in: walls/roof/windows (UA), air leakage, people (sensible + latent),
    lights, equipment, sun.
  * AC: rated capacity, COP that falls as the indoor/outdoor gap grows,
    fixed-speed thermostat (with 3-min compressor protection) or inverter.
  * The device only changes what an IR remote can change (on/off, setpoint,
    mode). The AC's own thermostat does the rest, as in real life.

Default inputs are PLACEHOLDERS (an 80-seat lecture room with 3 x 24,000 BTU
split ACs, synthetic Colombo weather and timetable). Replace them with your
measured data:
  --occupancy-csv  file with columns: minute,count   (minute 0 = day 1 00:00)
  --weather-csv    file with columns: hour,t_out,rh_out (rh in %)
  --measured-baseline-kwh  your metered kWh for the same period, to check
                           calibration (aim for within 10-15%; tune --ua,
                           --cop, --cap-kw until it matches).

Run:  python ac_sim.py                 (30 days, writes results/ folder)
      python ac_sim.py --days 7 --ac-type inverter --seats 40
Needs: numpy, matplotlib
"""

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass

import numpy as np

P_ATM = 101325.0      # Pa
H_FG = 2.45e6         # J/kg, latent heat of water
CP_AIR = 1005.0       # J/kg.K
RHO_AIR = 1.2         # kg/m3


# ---------------------------------------------------------------- psychrometrics
def p_ws(t):
    """Saturation vapour pressure (Pa) at t (C)."""
    return 610.94 * math.exp(17.625 * t / (t + 243.04))


def w_from_rh(t, rh):
    pw = rh * p_ws(t)
    return 0.622 * pw / (P_ATM - pw)


def rh_from_w(t, w):
    pw = w * P_ATM / (0.622 + w)
    return min(pw / p_ws(t), 1.0)


# ---------------------------------------------------------------- parameters
@dataclass
class Params:
    seats: int = 80
    floor_area: float = 120.0       # m2
    height: float = 3.2             # m
    ua_env: float = 300.0           # W/K walls+roof+windows
    ach: float = 0.7                # air changes per hour (leakage + door)
    c_air_mult: float = 3.0         # air + light furnishings
    c_mass: float = 15e6            # J/K building mass
    h_mass: float = 1200.0          # W/K air <-> mass
    lights_w: float = 1200.0
    equip_w: float = 400.0
    solar_peak_w: float = 1500.0
    sens_person: float = 75.0       # W sensible per seated person
    lat_person: float = 55.0        # W latent per person
    n_ac: int = 3
    cap_kw: float = 7.0             # rated cooling per AC (24,000 BTU ~ 7 kW)
    cop_rated: float = 3.2          # at 27 C in / 35 C out
    shr: float = 0.75               # sensible heat ratio in Cool mode at 55% RH
    fan_w: float = 60.0             # indoor fan per AC
    deadband: float = 0.5           # AC thermostat +/- C
    min_on: int = 3                 # compressor protection, minutes
    min_off: int = 3
    ac_type: str = "fixed"          # fixed | inverter
    device_w: float = 1.5           # our node's own draw, per AC
    tariff: float = 51.0            # LKR/kWh
    ef: float = 0.71                # kg CO2e/kWh
    # comfort band used by the device
    band_low: float = 22.0
    band_high: float = 26.0
    start_sp: float = 24.0
    rh_high: float = 0.65
    rh_ok: float = 0.60
    empty_off_min: int = 10
    step_every_min: int = 15
    override_hold_min: int = 45
    precool_at: tuple = (8 * 60, 13 * 60)   # minutes of day; () to disable

    @property
    def volume(self):
        return self.floor_area * self.height


# ---------------------------------------------------------------- inputs
def synth_weather(days, rng):
    """Hourly Colombo-like weather: ~25-32 C, humid, some rainy days."""
    t, rh = [], []
    for d in range(days):
        rainy = rng.random() < 0.3
        mean = 28.5 + rng.gauss(0, 0.6) - (1.2 if rainy else 0)
        amp = 2.5 if rainy else 3.5
        for h in range(24):
            c = math.cos(2 * math.pi * (h - 14) / 24)
            t.append(mean + amp * c)
            base = 0.86 if rainy else 0.78
            rh.append(min(0.97, max(0.5, base - 0.12 * c)))
    return np.array(t), np.array(rh)


def synth_occupancy(days, seats, rng):
    """Minute-by-minute headcount for a university lecture room."""
    occ = np.zeros(days * 1440, dtype=int)
    for d in range(days):
        weekday = d % 7 < 5
        slots = [8, 9, 10, 11, 13, 14, 15, 16] if weekday else [9, 10]
        p_class = 0.7 if weekday else 0.15
        for s in slots:
            if rng.random() > p_class:
                # empty slot: a few students may still sit and study
                if rng.random() < 0.3:
                    n = rng.randint(1, 4)
                    a = d * 1440 + s * 60 + rng.randint(0, 30)
                    L = rng.randint(15, 50)
                    occ[a:a + L] = np.maximum(occ[a:a + L], n)
                continue
            n = int(seats * rng.uniform(0.25, 1.0))
            a = d * 1440 + s * 60 + 5
            b = d * 1440 + s * 60 + 50
            occ[a:b] = n
            # some leave early (half the room empties)
            if rng.random() < 0.25:
                occ[a + 30:b] = n // 2
    return occ


def load_occupancy(path, n_min):
    occ = np.zeros(n_min, dtype=int)
    with open(path) as f:
        for r in csv.DictReader(f):
            m = int(r["minute"])
            if 0 <= m < n_min:
                occ[m] = int(float(r["count"]))
    return occ


def load_weather(path, days):
    t, rh = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            t.append(float(r["t_out"]))
            rh.append(float(r["rh_out"]) / 100.0)
    t, rh = np.array(t), np.array(rh)
    need = days * 24
    if len(t) < need:
        reps = need // len(t) + 1
        t, rh = np.tile(t, reps), np.tile(rh, reps)
    return t[:need], rh[:need]


# ---------------------------------------------------------------- controllers
class BaselineController:
    """Today's behaviour: staff switch on at 08:00 at a cold setpoint and off at
    17:30. Outside those hours the AC goes on when people arrive and is
    forgotten for 2 h after the last one leaves."""
    name = "baseline"

    def __init__(self, rng, sp_choices):
        self.rng = rng
        self.sp_choices = sp_choices
        self.day_sp = {}
        self.last_occ = -10_000

    def step(self, m, occupied, t, rh):
        d, mod = divmod(m, 1440)
        if d not in self.day_sp:
            vals, w = zip(*self.sp_choices)
            self.day_sp[d] = self.rng.choices(vals, weights=w)[0]
        if occupied:
            self.last_occ = m
        weekday = d % 7 < 5
        on = (weekday and 8 * 60 <= mod < 17 * 60 + 30) or (m - self.last_occ < 120)
        return on, self.day_sp[d], "cool"


class SmartController:
    """The device: radar presence + temp/humidity + IR. Only does what a remote can.

    Empty 10 min -> 27 C setback; empty 30 min -> off. On arrival, boost to
    22 C until the room is near 24 C, then work inside the 22-26 C band."""
    name = "smart"

    def __init__(self, p, rng):
        self.p, self.rng = p, rng
        self.on, self.sp, self.mode = False, p.start_sp, "cool"
        self.absent = 10_000
        self.last_eval = 0
        self.override_until = -1
        self.override_sp = None
        self.uncomfortable = 0
        self.overrides = 0
        self.boost = False
        self.setback = False

    def step(self, m, occupied, t, rh):
        p = self.p
        self.absent = 0 if occupied else self.absent + 1

        # Empty 10 min -> set back to 27 C (people may be between classes);
        # empty 30 min -> off.
        d, mod = divmod(m, 1440)
        if (not occupied and d % 7 < 5
                and any(s - 15 <= mod < s for s in p.precool_at)):
            # site schedule set on the dashboard: pre-cool 15 min before
            # the day's first class and after lunch
            self.on, self.sp, self.mode, self.last_eval = True, p.start_sp, "cool", m
            self.absent, self.setback = 0, False
            return True, p.start_sp, "cool"
        if self.absent >= p.empty_off_min + 20:
            self.on, self.sp, self.mode = False, p.start_sp, "cool"
            self.override_until, self.override_sp = -1, None
            self.setback = False
            return self.on, self.sp, self.mode
        if self.absent >= p.empty_off_min:
            self.override_until, self.override_sp = -1, None
            self.setback = True
            return True, 27.0, "cool"

        if not self.on or self.setback:
            self.setback = False
            # people arrived: boost-cool to 22 C until the room reaches 25 C
            self.on, self.mode, self.last_eval = True, "cool", m
            self.boost = t > p.start_sp + 1
            self.sp = p.band_low if self.boost else p.start_sp
        if getattr(self, "boost", False):
            if t <= p.start_sp + 1:
                self.boost, self.sp, self.last_eval = False, p.start_sp, m
            else:
                return True, p.band_low, "cool"

        # People feeling warm may grab the remote (IR receiver detects it)
        if occupied and (t > p.band_high + 0.5 or rh > 0.70):
            self.uncomfortable += 1
        else:
            self.uncomfortable = 0
        if self.uncomfortable >= 10 and m >= self.override_until and self.rng.random() < 0.3:
            self.override_sp = self.rng.choice([20.0, 22.0, 23.0])
            self.override_until = m + p.override_hold_min
            self.overrides += 1
            self.uncomfortable = 0

        if m < self.override_until:
            return True, self.override_sp, "cool"
        if self.override_sp is not None:          # grace over: ease back into band
            self.sp = max(p.band_low, min(p.band_high, self.override_sp))
            self.override_sp = None
            self.last_eval = m

        # Every 15 min: too warm -> step down; humid -> Dry/step down;
        # comfortable -> step up (max 26 C)
        if m - self.last_eval >= p.step_every_min:
            self.last_eval = m
            if t > self.sp + 1.0 or t > p.band_high + 0.5:
                self.mode = "cool"
                self.sp = max(p.band_low, self.sp - 1)
            elif rh > p.rh_high:
                if self.mode == "cool" and t <= self.sp + p.deadband:
                    self.mode = "dry"
                else:
                    self.mode = "cool"
                    self.sp = max(p.band_low, self.sp - 1)
            elif rh < p.rh_ok and t <= self.sp + p.deadband:
                self.mode = "cool"
                self.sp = min(p.band_high, self.sp + 1)
        return True, self.sp, self.mode


# ---------------------------------------------------------------- AC + room
def cop_at(p, t_in, t_out):
    def carnot(ti, to):
        te, tc = ti - 12 + 273.15, to + 12 + 273.15
        return te / max(tc - te, 5.0)
    eta = p.cop_rated / carnot(27.0, 35.0)
    return eta * carnot(t_in, t_out)


def simulate(p, ctrl, occ, w_t, w_rh, smart):
    n = len(occ)
    v = p.volume
    m_air = v * RHO_AIR
    c_air = m_air * CP_AIR * p.c_air_mult
    g_inf = p.ach * v * RHO_AIR * CP_AIR / 3600.0
    mdot_inf = p.ach * v * RHO_AIR / 3600.0
    w_min = w_from_rh(12.0, 0.95)       # coil can't dry air below this

    T, Tm = 28.0, 28.0
    W = w_from_rh(28.0, 0.70)
    comp, timer, integ = False, 99, 0.0

    out = {k: np.zeros(n) for k in ("T", "RH", "P", "sp", "on")}
    starts = 0
    alarm_min = 0

    for m in range(n):
        h = m // 60
        t_out, rh_out = w_t[h], w_rh[h]
        w_out = w_from_rh(t_out, rh_out)
        rh = rh_from_w(T, W)
        people = occ[m]

        on, sp, mode = ctrl.step(m, people > 0, T, rh)

        # --- AC internal thermostat
        timer += 1
        frac = 0.0
        if not on:
            if comp:
                comp, timer = False, 0
            integ = 0.0
        elif p.ac_type == "fixed":
            if comp and T < sp - p.deadband and timer >= p.min_on:
                comp, timer = False, 0
            elif not comp and T > sp + p.deadband and timer >= p.min_off:
                comp, timer, starts = True, 0, starts + 1
            frac = 1.0 if comp else 0.0
        else:  # inverter
            err = T - sp
            integ = max(-30.0, min(30.0, integ + err))
            f = max(0.0, min(1.0, 0.55 + 0.4 * err + 0.03 * integ))
            if comp and f < 0.25 and T < sp - p.deadband and timer >= p.min_on:
                comp, timer = False, 0
            elif not comp and err > p.deadband and timer >= p.min_off:
                comp, timer, starts = True, 0, starts + 1
            frac = max(f, 0.25) if comp else 0.0

        cap = p.n_ac * p.cap_kw * 1000 * (1 + 0.01 * (T - 27)) * (1 - 0.008 * (t_out - 35))
        # coil removes more moisture when the room is humid (SHR falls with RH)
        shr = max(0.55, min(0.95, p.shr - 1.1 * (rh - 0.55)))
        if mode == "dry":
            cap *= 0.6
            shr = max(0.4, shr - 0.2)
        q_tot = cap * frac
        q_lat = q_tot * (1 - shr)
        if W <= w_min:
            q_lat = 0.0
        q_sens = q_tot - q_lat

        cop = cop_at(p, T, t_out)
        if p.ac_type == "inverter" and frac > 0:
            cop *= 1 + 0.25 * (1 - frac)
        power = q_tot / cop if q_tot > 0 else 0.0
        if on:
            power += p.fan_w * p.n_ac
        if smart:
            power += p.device_w * p.n_ac

        # --- heat gains
        hod = (m % 1440) / 60.0
        sun = p.solar_peak_w * max(0.0, math.sin(math.pi * (hod - 6.5) / 12)) if 6.5 < hod < 18.5 else 0.0
        lights = p.lights_w if people > 0 else 0.0
        internal = people * p.sens_person + lights + p.equip_w
        q_mass = p.h_mass * (Tm - T)
        dT = (p.ua_env * (t_out - T) + g_inf * (t_out - T) + q_mass + internal + 0.5 * sun - q_sens) / c_air
        T += 60 * dT
        Tm += 60 * (p.h_mass * (T - Tm) + 0.5 * sun) / p.c_mass

        dW = (mdot_inf * (w_out - W) + people * p.lat_person / H_FG - q_lat / H_FG) / m_air
        W = max(0.004, W + 60 * dW)

        if on and people > 0 and T > 28.0:
            alarm_min += 1

        out["T"][m], out["RH"][m], out["P"][m] = T, rh_from_w(T, W), power
        out["sp"][m], out["on"][m] = sp if on else np.nan, on

    out["starts"], out["alarm_min"] = starts, alarm_min
    return out


# ---------------------------------------------------------------- report
def summarise(p, r, occ, days):
    kwh = r["P"].sum() / 60_000
    occupied = occ > 0
    T, RH = r["T"][occupied], r["RH"][occupied]
    nocc = max(occupied.sum(), 1)
    comfy = ((T >= p.band_low) & (T <= p.band_high + 0.5) & (RH <= 0.70)).sum()
    return {
        "kWh": kwh,
        "kWh_per_day": kwh / days,
        "LKR": kwh * p.tariff,
        "kgCO2e": kwh * p.ef,
        "AC_on_hours": r["on"].sum() / 60,
        "AC_on_while_empty_hours": (r["on"].astype(bool) & ~occupied).sum() / 60,
        "comfort_pct": 100 * comfy / nocc,
        "too_cold_pct": 100 * (T < p.band_low).sum() / nocc,
        "too_warm_pct": 100 * (T > p.band_high + 0.5).sum() / nocc,
        "mean_T_occupied": float(T.mean()) if len(T) else float("nan"),
        "mean_RH_occupied_pct": float(100 * RH.mean()) if len(RH) else float("nan"),
        "compressor_starts": r["starts"],
        "alarm_minutes_over_28C": r["alarm_min"],
    }


def plot(p, base, smart, occ, w_t, days, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    green, orange, grey = "#1B7A5A", "#E9A23B", "#8A949C"

    # pick the busiest weekday for the profile plot
    weekdays = [d for d in range(days) if d % 7 < 5] or [0]
    d = max(weekdays, key=lambda d: occ[d * 1440:(d + 1) * 1440].sum())
    sl = slice(d * 1440 + 6 * 60, d * 1440 + 20 * 60)
    x = np.arange(6 * 60, 20 * 60) / 60

    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(x, base["T"][sl], color=orange, label="Room temp, baseline")
    ax[0].plot(x, smart["T"][sl], color=green, label="Room temp, with device")
    ax[0].plot(x, w_t[d * 24 + 6:d * 24 + 20].repeat(60), color=grey, ls=":", label="Outdoor")
    ax[0].axhspan(p.band_low, p.band_high, color=green, alpha=0.08)
    ax[0].set_ylabel("°C")
    ax[0].legend(loc="upper left", fontsize=8, frameon=False)
    ax[0].set_title(f"Day {d + 1}: busiest weekday", loc="left", fontsize=11)

    ax[1].plot(x, base["P"][sl] / 1000, color=orange, lw=0.8, label="Baseline")
    ax[1].plot(x, smart["P"][sl] / 1000, color=green, lw=0.8, label="With device")
    ax[1].set_ylabel("AC power, kW")
    ax[1].legend(loc="upper left", fontsize=8, frameon=False)

    ax[2].fill_between(x, occ[sl], step="post", color=grey, alpha=0.5)
    ax[2].set_ylabel("People")
    ax[2].set_xlabel("Hour of day")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "day_profile.png"), dpi=150)
    plt.close(fig)

    bd = base["P"].reshape(days, 1440).sum(1) / 60_000
    sd = smart["P"].reshape(days, 1440).sum(1) / 60_000
    fig, ax = plt.subplots(figsize=(11, 4))
    i = np.arange(1, days + 1)
    ax.bar(i - 0.2, bd, 0.4, color=orange, label="Baseline")
    ax.bar(i + 0.2, sd, 0.4, color=green, label="With device")
    ax.set_xlabel("Day")
    ax.set_ylabel("kWh per day")
    ax.set_title(f"Daily AC energy: {100 * (1 - sd.sum() / bd.sum()):.0f}% saved over {days} days",
                 loc="left", fontsize=11)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "daily_energy.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seats", type=int, default=80)
    ap.add_argument("--n-ac", type=int, default=3)
    ap.add_argument("--cap-kw", type=float, default=7.0, help="cooling per AC, kW")
    ap.add_argument("--cop", type=float, default=3.2)
    ap.add_argument("--ua", type=float, default=300.0, help="envelope W/K")
    ap.add_argument("--ac-type", choices=["fixed", "inverter"], default="fixed")
    ap.add_argument("--tariff", type=float, default=51.0, help="LKR per kWh")
    ap.add_argument("--baseline-sp", default="18:0.3,20:0.4,22:0.2,24:0.1",
                    help="baseline setpoint mix, e.g. '18:0.3,20:0.4,22:0.3'")
    ap.add_argument("--occupancy-csv")
    ap.add_argument("--weather-csv")
    ap.add_argument("--measured-baseline-kwh", type=float)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    p = Params(seats=a.seats, n_ac=a.n_ac, cap_kw=a.cap_kw, cop_rated=a.cop,
               ua_env=a.ua, ac_type=a.ac_type, tariff=a.tariff)
    rng = random.Random(a.seed)
    n_min = a.days * 1440

    w_t, w_rh = load_weather(a.weather_csv, a.days) if a.weather_csv else synth_weather(a.days, rng)
    occ = load_occupancy(a.occupancy_csv, n_min) if a.occupancy_csv else synth_occupancy(a.days, a.seats, rng)
    sp_mix = [(float(k), float(v)) for k, v in (s.split(":") for s in a.baseline_sp.split(","))]

    base = simulate(p, BaselineController(random.Random(a.seed + 1), sp_mix), occ, w_t, w_rh, smart=False)
    sc = SmartController(p, random.Random(a.seed + 2))
    smart = simulate(p, sc, occ, w_t, w_rh, smart=True)

    sb, ss = summarise(p, base, occ, a.days), summarise(p, smart, occ, a.days)
    ss["remote_overrides"] = sc.overrides
    sb["remote_overrides"] = 0

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["metric", "baseline", "with_device"])
        for k in sb:
            wr.writerow([k, round(sb[k], 2), round(ss[k], 2)])
    plot(p, base, smart, occ, w_t, a.days, a.out)

    saved = sb["kWh"] - ss["kWh"]
    pct = 100 * saved / sb["kWh"]
    print(f"\n=== {a.days}-day simulation | {a.seats} seats | {p.n_ac} x {p.cap_kw} kW {p.ac_type} AC ===")
    print(f"{'metric':32}{'baseline':>12}{'with device':>14}")
    for k in sb:
        print(f"{k:32}{sb[k]:>12.1f}{ss[k]:>14.1f}")
    print(f"\nEnergy saved: {saved:.0f} kWh ({pct:.0f}%)")
    print(f"Money saved:  LKR {saved * p.tariff:,.0f}  (at LKR {p.tariff}/kWh)")
    print(f"CO2e avoided: {saved * p.ef:.0f} kg")
    per_ac_month = saved * p.tariff / p.n_ac * 30 / a.days
    print(f"Per AC per month: LKR {per_ac_month:,.0f} -> payback on LKR 17,500 in "
          f"{17_500 / per_ac_month:.1f} months" if per_ac_month > 0 else "")
    if a.measured_baseline_kwh:
        err = 100 * (sb["kWh"] - a.measured_baseline_kwh) / a.measured_baseline_kwh
        print(f"\nCalibration: simulated baseline {sb['kWh']:.0f} kWh vs measured "
              f"{a.measured_baseline_kwh:.0f} kWh ({err:+.0f}%). "
              f"{'OK' if abs(err) <= 15 else 'Tune --ua / --cop / --cap-kw until within 15%.'}")
    print(f"\nWrote {a.out}/summary.csv, day_profile.png, daily_energy.png")


if __name__ == "__main__":
    main()
