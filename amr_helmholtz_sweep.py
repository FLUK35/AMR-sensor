#!/usr/bin/env python
"""
AMR sensor characterisation in a Helmholtz coil.

Hardware
--------
  * EA-PS 2042-10B  (42 V / 10 A)  -> drives the Helmholtz coil   (eaps2000 lib, USB/serial)
  * Keithley DMM/SMU               -> precise DC voltage reading   (pyvisa, GPIB/USB/RS-232)
  * NI myDAQ                       -> analog inputs (+ optional DO for a polarity relay) (nidaqmx)
                                      ai0 = bridge supply, ai1 = voltage over the coil-current shunt
  * Shunt resistor (0.1 ohm) in the coil circuit, low side: PSU minus terminal -> shunt -> coil
    leads (swapped here for a manual bipolar sweep). It never moves during a swap and its common
    mode stays near 0 V; the myDAQ inputs only take +/-10 V common mode.
  * Oscillaboard with the AMR sensor, placed in the centre of the coil

Coil current: the myDAQ shunt measurement is the standard
---------------------------------------------------------
  The coil current used for the field is the current MEASURED over the shunt by the myDAQ
  (current_source = "shunt", the default). It is the real current in the coil, averaged over
  daq_samples samples. The EA-PS read-back is coarse (it reads 0 below ~0.05 A and scatters by
  ~10 mA) and is only stored for reference and used as a cross-check. current_source = "psu"
  (--current-source psu) falls back to the PSU read-back; --no-daq does that automatically.

What it does
------------
  1. Configures the PSU (OVP/OCP, compliance limit), output OFF.
  2. Sweeps the coil current (or voltage) 0 -> +max -> 0 (optionally -> -max -> 0, reversing the
     coil by hand or with a polarity relay driven by the myDAQ), ramping gently between points
     (the coil is inductive).
  3. At every point: waits to settle, reads PSU actual V/I, Keithley voltage, myDAQ channels.
  4. Computes the field in the coil centre:  B = (4/5)^(3/2) * mu0 * N * I / R  from the shunt
     current (B_shunt_mT, the standard), plus the PSU read-back (B_mT) and the setpoint (B_set_mT).
  5. Writes every point to CSV immediately (flushed, so nothing is lost on a crash/Ctrl+C),
     plus a JSON sidecar with all settings and instrument IDs, and a PNG plot at the end.
  6. ALWAYS ramps the coil down and switches the PSU output off on exit / error / Ctrl+C.

Usage
-----
  python amr_helmholtz_sweep.py --list                 # show COM ports, VISA resources, DAQ devices
  python amr_helmholtz_sweep.py --simulate             # dry run with fake instruments
  python amr_helmholtz_sweep.py --psu-port COM5 --keithley "GPIB0::16::INSTR" --max 2 --step 0.05
  python amr_helmholtz_sweep.py --bipolar              # +/- sweep, pauses so you can swap the coil leads
  python amr_helmholtz_sweep.py --no-keithley          # run without one of the instruments
  python amr_helmholtz_sweep.py --no-daq               # (no shunt: field from the PSU read-back)
  python amr_helmholtz_sweep.py --sr-test 10           # no coil current: check the set/reset pulses
  python amr_helmholtz_sweep.py --sr flip              # set+reset at every point, offset-free signal

Sensor: Honeywell HMC1021S (single axis)
----------------------------------------
  Typical datasheet values (check your datasheet revision): sensitivity ~1.0 mV/V/gauss,
  linear range +/-6 gauss (= +/-0.6 mT, ~0.87 A in this coil), bridge ~1.1 kOhm.
  Only the field along the sensitive axis (in the plane of the chip) counts; a field along
  the other in-plane axis (the easy axis) changes the sensitivity and, if strong, flips the
  magnetisation.

Set/reset
---------
  A strong field or a shock can scramble the sensor's magnetisation; a set (or reset) pulse
  through the on-chip S/R strap restores it. Reset flips the sign of the sensitivity but not the
  bridge offset, so (V_set - V_reset)/2 is the field signal with the offset and its drift removed.
  The HMC1021 strap needs a short pulse of roughly 0.5 A (check the datasheet). The myDAQ line
  (3.3 V, a few mA) CANNOT drive it: it only triggers an external pulse driver (e.g. the
  capacitor + complementary MOSFET circuit in the HMC1021/1022 datasheet) that fires one pulse
  per edge on its input.

Requirements:  pip install eaps2000 pyvisa nidaqmx matplotlib
  (+ NI-DAQmx driver and NI-VISA (or `pip install pyvisa-py`) on the PC)

File layout
-----------
  Config              all settings (dataclass) + coil constant + sanity checks
  PowerSupply         EA-PS wrapper: setup, slew-limited ramp, read back, safe close
  Keithley            SCPI voltmeter wrapper (DMM or SMU)
  MyDAQ               analog inputs (incl. the shunt) + digital lines for relay and S/R driver
  _SimWorld, Sim*     fake instruments for --simulate
  Rig                 the connected instruments + coil polarity; connect / close in one place
  build_setpoints     list of setpoints for one loop
  csv_header          CSV column names for the enabled channels
  read_sensors,
  measure_sensors     sensor readings at one point (incl. set/reset handling)
  measured_current    coil current from the standard source (shunt, or PSU read-back)
  set_polarity,
  wait_for_manual_swap reverse the coil (relay or by hand) at zero current
  measure_point       one full point: polarity, ramp, settle, read, CSV row
  run                 the measurement itself (connect, sweep, write files, shut down)
  sr_test             check the set/reset pulses without coil current
  make_plot           quick PNG of every signal vs B
  build_parser,
  apply_args, main    command line -> Config -> run

Note: the instrument libraries (eaps2000, pyvisa, nidaqmx) are imported inside the classes,
not at the top of the file. That way --simulate and --list work on a PC where some of the
drivers are not installed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

MU0 = 4e-7 * math.pi  # T*m/A  (vacuum permeability)


# =====================================================================================
#  USER SETTINGS  (edit here, or override the common ones from the command line)
# =====================================================================================
@dataclass
class Config:
    """All settings of one sweep.

    The defaults below are used unless main() overrides them from the command line.
    The whole object is stored in the JSON file of every run (via dataclasses.asdict), so
    the settings of each measurement can always be looked up afterwards.
    """
    # ---- EA-PS 2042-10B (nominal 42 V / 10 A) ----
    psu_port: str = "COM3"            # see --list ; e.g. "COM5"
    psu_ovp: float = 20.0             # V  over-voltage protection (PSU switches off above this)
    psu_ocp: float = 3.0              # A  over-current protection (PSU switches off above this)
    psu_voltage_limit: float = 15.0   # V  compliance voltage used in "current" sweep mode
    psu_current_limit: float = 2.5    # A  compliance current used in "voltage" sweep mode

    # ---- Sweep ----
    sweep_mode: str = "current"       # "current" (CC, recommended for a coil: B ~ I) or "voltage"
    sweep_max: float = 2.0            # A (or V in voltage mode)
    sweep_step: float = 0.05          # A (or V)
    bipolar: bool = True              # True -> also sweep negative (relay or manual lead swap)
                                      # NOTE: main() replaces this with --bipolar
    polarity_switch: str = "manual"   # how the coil polarity is reversed in a bipolar sweep:
                                      # "manual" pause, output off, you swap the coil leads and confirm
                                      # "relay"  myDAQ DO line (polarity_line) drives a relay
    loops: int = 1                    # repeat full loop (useful for hysteresis/repeatability)
    settle_time: float = 0.5          # s  wait after reaching each setpoint
    ramp_rate: float = 1.0            # A/s (or V/s) max slew between setpoints

    # ---- Helmholtz coil (per coil: N turns, radius R, coil spacing = R) ----
    coil_turns: int = 154             # turns per coil (11 x 14 winding)
    coil_radius_m: float = 0.20       # m  (diameter 0.40 m)

    # ---- Coil current: which measurement is the standard ----
    current_source: str = "shunt"     # "shunt" myDAQ voltage over the shunt (default, recommended)
                                      # "psu"   EA-PS read-back (coarse; only without a shunt)

    # ---- Keithley ----
    use_keithley: bool = True         # NOTE: main() replaces this with "not --no-keithley"
    keithley_resource: str = "GPIB0::16::INSTR"   # see --list ; USB0::0x05E6::...::INSTR, ASRL4::INSTR
    keithley_type: str = "smu"        # "dmm" (2000/2001/2010/2100/DMM6500 SCPI)  or "smu" (2400 series)
    keithley_nplc: float = 1.0        # integration time in power-line cycles (higher = less noise)
    keithley_n_avg: int = 3           # readings averaged per point
    keithley_label: str = "V_keithley"  # what it measures, e.g. AMR bridge output (CSV column name)
    keithley_baud: int = 9600         # only used for ASRL (RS-232) resources

    # ---- NI myDAQ ----
    use_daq: bool = True
    daq_device: str = "myDAQ1"        # name in NI MAX
    daq_channels: tuple = ("ai0", "ai1")
    daq_labels: tuple = ("V_bridge_supply", "V_shunt")  # CSV column names, same order as channels
    daq_ranges: tuple = (10.0, 2.0)   # V, input range per channel (myDAQ: 10 or 2)
    daq_rate: float = 10000.0         # S/s per channel (myDAQ max 200 kS/s total)
    daq_samples: int = 1000           # samples per point (averaged; std is stored too)
    polarity_line: str = "port0/line0"  # myDAQ DO line driving the polarity relay (bipolar only)

    # ---- Coil-current shunt, read by one of the myDAQ inputs ----
    shunt_label: str = "V_shunt"      # the daq_labels entry wired across the shunt ("" = no shunt)
    shunt_ohm: float = 0.1            # ohm
    shunt_position: str = "psu"       # "psu":  between the PSU minus terminal and the lead swap/relay.
                                      #         V_shunt is always >= 0; the polarity gives the sign.
                                      # "coil": in series with the coil after the swap/relay.
                                      #         V_shunt changes sign with the polarity by itself.

    # ---- HMC1021S set/reset (S/R) pulses, via an external pulse driver on a myDAQ DO line ----
    sr_mode: str = "off"              # "off"   no pulses
                                      # "once"  one set pulse before the sweep
                                      # "point" a set pulse before every reading
                                      # "flip"  every point: set -> read, reset -> read; also
                                      #         writes <label>_reset and <label>_sr = (set-reset)/2
    sr_line: str = "port0/line1"      # myDAQ DO line feeding the pulse driver input
    sr_set_on_rising: bool = True     # True: rising edge = set pulse, falling = reset (driver-dependent)
    sr_hold: float = 0.05             # s  wait after each edge (driver recharge + bridge settling)

    # ---- Output ----
    out_dir: str = "data"             # folder for CSV / JSON / PNG (created if missing)
    sample_name: str = "AMR"          # first part of the file names
    notes: str = ""                   # free text, stored in the JSON

    # ---- derived quantities -----------------------------------------------------------
    def coil_constant_T_per_A(self) -> float:
        """Field at the centre of the Helmholtz pair per ampere, in tesla per ampere.

        B = (4/5)^(3/2) * mu0 * N * I / R  for two coils of N turns and radius R spaced R apart.
        With N = 154 and R = 0.20 m this is 0.6924 mT/A.
        """
        return (4.0 / 5.0) ** 1.5 * MU0 * self.coil_turns / self.coil_radius_m

    def field_mT(self, current_A: float) -> float:
        """Field at the coil centre (mT) for a coil current (A)."""
        return current_A * self.coil_constant_T_per_A() * 1e3

    def has_shunt(self) -> bool:
        """True if the coil current is measured over the shunt (myDAQ in use, label present)."""
        return self.use_daq and bool(self.shunt_label) and self.shunt_label in self.daq_labels

    def is_shunt(self, label: str) -> bool:
        """True if `label` is the myDAQ channel wired across the shunt (not a sensor signal)."""
        return self.has_shunt() and label == self.shunt_label

    def shunt_current(self, v_shunt: float, polarity: int) -> float:
        """Signed coil current (A) from the voltage over the shunt."""
        i = v_shunt / self.shunt_ohm
        return polarity * i if self.shunt_position == "psu" else i

    def needs_daq(self) -> bool:
        """The myDAQ is needed for its inputs, the polarity relay, or the set/reset line."""
        relay = self.bipolar and self.polarity_switch == "relay"
        return self.use_daq or relay or self.sr_mode != "off"

    # ---- sanity checks ------------------------------------------------------------------
    def validate(self):
        """Refuse settings that are invalid or could damage something, BEFORE connecting.

        The protection limits (OVP/OCP) must sit above the values the sweep will actually
        use, otherwise the PSU would trip in the middle of the sweep. Raises ValueError
        (not assert, which `python -O` would silently skip).
        """
        def need(ok: bool, msg: str):
            if not ok:
                raise ValueError(msg)

        need(self.sweep_mode in ("current", "voltage"), "sweep_mode must be 'current' or 'voltage'")
        need(self.keithley_type in ("dmm", "smu"), "keithley_type must be 'dmm' or 'smu'")
        need(len(self.daq_channels) == len(self.daq_labels) == len(self.daq_ranges),
             "daq_channels/daq_labels/daq_ranges length mismatch")
        need(all(r in (2.0, 10.0) for r in self.daq_ranges), "myDAQ input ranges are 2 or 10 V")
        need(0 < self.sweep_step <= self.sweep_max, "need 0 < sweep_step <= sweep_max")
        need(self.psu_voltage_limit <= 42.0 and self.psu_current_limit <= 10.0,
             "exceeds PS 2042-10B rating")
        need(self.sr_mode in ("off", "once", "point", "flip"), "sr_mode must be off/once/point/flip")
        need(self.polarity_switch in ("manual", "relay"), "polarity_switch must be 'manual' or 'relay'")

        # The coil current: the shunt is the standard, so it must actually be there.
        need(self.current_source in ("shunt", "psu"), "current_source must be 'shunt' or 'psu'")
        if self.current_source == "shunt":
            why = ("the myDAQ inputs are off (--no-daq / use_daq=False)" if not self.use_daq else
                   f"shunt_label {self.shunt_label!r} is not one of daq_labels {self.daq_labels}")
            need(self.has_shunt(), f"current_source 'shunt' needs the shunt reading, but {why}; "
                                   f"use --current-source psu to run on the PSU read-back")
        if self.has_shunt():
            need(self.shunt_ohm > 0, "shunt_ohm must be > 0")
            need(self.shunt_position in ("psu", "coil"), "shunt_position must be 'psu' or 'coil'")
            # The largest current the PSU can deliver must not clip the shunt channel.
            i_max = self.sweep_max if self.sweep_mode == "current" else self.psu_current_limit
            rng = self.daq_ranges[self.daq_labels.index(self.shunt_label)]
            need(i_max * self.shunt_ohm < 0.95 * rng,
                 f"{i_max} A x {self.shunt_ohm} ohm = {i_max * self.shunt_ohm:.2f} V exceeds the "
                 f"{rng} V range of {self.shunt_label}: use the 10 V range or a smaller shunt")

        if self.sr_mode != "off" and self.bipolar and self.polarity_switch == "relay":
            need(self.sr_line != self.polarity_line, "sr_line and polarity_line must be different DO lines")
        if self.sr_mode == "flip":
            need(self.use_keithley or self.use_daq, "sr_mode 'flip' needs the Keithley or the myDAQ inputs")
        if self.sweep_mode == "current":
            need(self.sweep_max < self.psu_ocp,
                 f"sweep_max {self.sweep_max} A must be below OCP {self.psu_ocp} A")
            need(self.psu_voltage_limit < self.psu_ovp, "psu_voltage_limit must be below OVP")
        else:
            need(self.sweep_max < self.psu_ovp, "sweep_max (V) must be below OVP")
            need(self.psu_current_limit < self.psu_ocp, "psu_current_limit must be below OCP")


# =====================================================================================
#  Instruments
# =====================================================================================
class PowerSupply:
    """EA-PS 2000 B wrapper (unipolar: polarity reversal needs an external relay).

    In "current" mode the PSU regulates the current (CC mode) and the voltage limit is only
    a ceiling. That is what we want for a coil: the field is proportional to the current,
    and the current does not change when the coil warms up and its resistance rises.
    `level` remembers the present setpoint, because ramp_to() moves from there.
    """

    def __init__(self, cfg: Config):
        from eaps2000 import eaps2k
        self.cfg = cfg
        self.dev = eaps2k(cfg.psu_port)
        self.dev.set_remote(True)     # take control from the front panel
        self.level = 0.0  # present setpoint of the swept quantity

    def idn(self) -> str:
        """Identification string (maker, type, serial number, firmware) for the JSON file."""
        return f"{self.dev.get_manufacturer()} {self.dev.get_type()} SN {self.dev.get_serial()} FW {self.dev.get_version()}"

    def setup(self):
        """Put the PSU in a known, safe state: output off, alarms cleared, limits set, level 0."""
        c = self.cfg
        self.dev.set_output_state(False)
        self.dev.ack_alarm()          # clear any old protection alarm
        self.dev.set_ovp(c.psu_ovp)
        self.dev.set_ocp(c.psu_ocp)
        if c.sweep_mode == "current":
            self.dev.set_voltage(c.psu_voltage_limit)  # compliance
            self.dev.set_current(0.0)
        else:
            self.dev.set_current(c.psu_current_limit)  # compliance
            self.dev.set_voltage(0.0)
        self.level = 0.0

    def _set(self, x: float):
        """Set the swept quantity (current or voltage) directly, without ramping."""
        if self.cfg.sweep_mode == "current":
            self.dev.set_current(x)
        else:
            self.dev.set_voltage(x)
        self.level = x

    def ramp_to(self, target: float, dt: float = 0.1):
        """Slew-limited move to `target` (>= 0) so the coil inductance doesn't kick.

        A sudden current change in an inductor produces a voltage spike (V = L dI/dt).
        Instead the level moves in steps of at most ramp_rate * dt (0.1 A at 1 A/s),
        one step every dt seconds, and finally lands exactly on the target.
        """
        max_step = max(self.cfg.ramp_rate * dt, 1e-6)
        while abs(target - self.level) > max_step:
            # Move one step towards the target (copysign gives the step the right sign).
            self._set(self.level + math.copysign(max_step, target - self.level))
            time.sleep(dt)
        self._set(target)

    def output(self, on: bool):
        """Switch the PSU output on or off."""
        self.dev.set_output_state(on)

    def read(self) -> dict:
        """Read back the ACTUAL output (not the setpoint) and the status flags.

        mode is "CC" (regulating current) or "CV" (regulating voltage). The four flags are the
        protections: over-voltage, over-current, over-power, over-temperature.
        """
        a = self.dev.get_actual()
        return {"V": a["V"], "I": a["I"], "mode": "CC" if a["CC"] else "CV",
                "OVP": a["OVP"], "OCP": a["OCP"], "OPP": a["OPP"], "OTP": a["OTP"]}

    def close(self):
        """Ramp to zero and switch off. The `finally` makes sure the output is switched off
        and the port is released even if the ramp fails (e.g. the USB link dropped)."""
        try:
            self.ramp_to(0.0)
        finally:
            self.dev.set_output_state(False)
            self.dev.set_remote(False)   # give control back to the front panel
            self.dev.ser_dev.close()


# Matches the first number in a text, e.g. "+1.234567E-03VDC" -> "+1.234567E-03".
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _first_float(s: str) -> float:
    """Extract the first number from an instrument reply.

    Some Keithley models add units or extra fields (e.g. "+1.23E-3VDC,+12.3SECS"); this takes
    only the first value instead of failing on the extra text.
    """
    m = _FLOAT_RE.search(s)
    if not m:
        raise ValueError(f"no number in instrument reply {s!r}")
    return float(m.group())


def _mean_std(vals) -> tuple[float, float]:
    """Mean and sample standard deviation (0 for a single value)."""
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


class Keithley:
    """Generic SCPI Keithley used as a DC voltmeter (DMM, or SMU sourcing 0 A).

    An SMU sourcing 0 A while measuring voltage acts as a voltmeter with very high input
    impedance, so it does not load the sensor bridge.
    """

    def __init__(self, cfg: Config):
        import pyvisa
        self.cfg = cfg
        self.rm = pyvisa.ResourceManager()
        self.inst = self.rm.open_resource(cfg.keithley_resource)
        self.inst.timeout = 10000  # ms (long NPLC settings make a reading slow)
        self.inst.read_termination = "\n"
        self.inst.write_termination = "\n"
        self.is_gpib = cfg.keithley_resource.upper().startswith("GPIB")
        if cfg.keithley_resource.upper().startswith("ASRL"):
            # RS-232: the baud rate must match the setting on the instrument.
            self.inst.baud_rate = cfg.keithley_baud
        elif self.is_gpib:
            # GPIB device clear: aborts a reading left hanging by an earlier crashed run
            # and empties the instrument's input/output buffers.
            self.inst.clear()

    def idn(self) -> str:
        """Standard SCPI identification query."""
        return self.inst.query("*IDN?").strip()

    def setup(self):
        """Reset the instrument and configure it for DC voltage readings."""
        w, c = self.inst.write, self.cfg
        w("*RST")                     # factory defaults
        w("*CLS")                     # clear the error queue
        if c.keithley_type == "dmm":
            w(":SENS:FUNC 'VOLT:DC'")
            w(":SENS:VOLT:DC:RANG:AUTO ON")
            w(f":SENS:VOLT:DC:NPLC {c.keithley_nplc}")
            w(":INIT:CONT OFF")       # no free-running; one reading per :READ?
        else:  # 2400-series SMU as high-impedance voltmeter
            w(":SOUR:FUNC CURR")      # source current ...
            w(":SOUR:CURR:LEV 0")     # ... of 0 A
            w(":SENS:FUNC 'VOLT'")    # and measure voltage
            w(":SENS:VOLT:PROT 21")   # voltage compliance 21 V
            w(":SENS:VOLT:RANG:AUTO ON")
            w(f":SENS:VOLT:NPLC {c.keithley_nplc}")
            w(":FORM:ELEM VOLT")      # reply with the voltage only
            w(":OUTP ON")
        time.sleep(0.5)               # let the settings take effect

    def read(self) -> tuple[float, float]:
        """Take keithley_n_avg readings; return their mean and standard deviation (volts)."""
        return _mean_std([_first_float(self.inst.query(":READ?")) for _ in range(self.cfg.keithley_n_avg)])

    def close(self):
        """Switch the SMU output off, return the instrument to local (front panel) control."""
        try:
            if self.cfg.keithley_type == "smu":
                self.inst.write(":OUTP OFF")
            if self.is_gpib:
                # :SYST:LOC is an RS-232-only command; on GPIB the bus releases the
                # instrument to the front panel with "go to local" (GTL).
                from pyvisa.constants import RenLineOperation
                self.inst.control_ren(RenLineOperation.address_gtl)
            else:
                self.inst.write(":SYST:LOC")
        finally:
            self.inst.close()
            self.rm.close()


class MyDAQ:
    """NI myDAQ: finite, averaged differential AI reads + optional DO lines.

    One analog-input task reads all channels together, including the shunt that measures the
    coil current. Each read takes daq_samples samples per channel at daq_rate (1000 samples at
    10 kS/s = 0.1 s = exactly 5 cycles of 50 Hz mains), and returns the mean and standard
    deviation per channel. Averaging over whole mains cycles cancels most 50 Hz pickup in the
    mean; it still shows up in the std.
    """

    def __init__(self, cfg: Config):
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, TerminalConfiguration
        self.cfg = cfg
        # The constant is called DIFF in newer nidaqmx versions and DIFFERENTIAL in older ones.
        diff = getattr(TerminalConfiguration, "DIFF", None) or getattr(TerminalConfiguration, "DIFFERENTIAL")
        self.ai = nidaqmx.Task()
        for ch, rng in zip(cfg.daq_channels, cfg.daq_ranges):
            self.ai.ai_channels.add_ai_voltage_chan(
                f"{cfg.daq_device}/{ch}", terminal_config=diff, min_val=-rng, max_val=rng)
        # FINITE: each read() takes exactly daq_samples samples, then stops.
        self.ai.timing.cfg_samp_clk_timing(cfg.daq_rate, sample_mode=AcquisitionType.FINITE,
                                           samps_per_chan=cfg.daq_samples)
        # Digital outputs, all in ONE task (the lines share a port): "pol" drives the polarity
        # relay (bipolar sweeps, False = normal), "sr" triggers the set/reset pulse driver.
        # _do_state holds the present level of every line; _write_do() writes them all.
        self.do = None
        self._do_names, self._do_state = [], {}
        lines = []
        if cfg.bipolar and cfg.polarity_switch == "relay":
            lines.append(("pol", cfg.polarity_line))
        if cfg.sr_mode != "off":
            lines.append(("sr", cfg.sr_line))
        if lines:
            self.do = nidaqmx.Task()
            for name, line in lines:
                self.do.do_channels.add_do_chan(f"{cfg.daq_device}/{line}")
                self._do_names.append(name)
                self._do_state[name] = False
            self._write_do()

    def _write_do(self):
        """Write the present level of every DO line (one value per channel)."""
        vals = [self._do_state[n] for n in self._do_names]
        self.do.write(vals[0] if len(vals) == 1 else vals)

    def idn(self) -> str:
        """Product type and serial number (hex) for the JSON file."""
        import nidaqmx.system
        d = nidaqmx.system.Device(self.cfg.daq_device)
        return f"{d.product_type} SN {d.serial_num:X}"

    def read(self) -> list[tuple[float, float]]:
        """One (mean, std) pair in volts per channel, in the order of daq_channels."""
        n = self.cfg.daq_samples
        # Timeout = acquisition time plus 5 s margin.
        data = self.ai.read(number_of_samples_per_channel=n, timeout=n / self.cfg.daq_rate + 5)
        if len(self.cfg.daq_channels) == 1:
            data = [data]  # with one channel nidaqmx returns a flat list; make it a list of lists
        return [_mean_std(ch) for ch in data]

    def set_polarity(self, negative: bool):
        """Switch the polarity relay (True = reversed). Only call with the coil current at 0."""
        if "pol" not in self._do_state:
            raise RuntimeError("polarity relay not configured (bipolar=False or polarity_switch='manual')")
        self._do_state["pol"] = bool(negative)
        self._write_do()

    def sr_pulse(self, kind: str):
        """Fire a "set" or "reset" pulse through the sensor's S/R strap (via the driver).

        The driver turns an EDGE on sr_line into one short high-current pulse; which edge is
        "set" is given by sr_set_on_rising. If the line already sits at the level the wanted
        edge ends on, it first moves the other way (firing the opposite pulse), so the sensor
        always ends with a fresh pulse of the requested kind.
        """
        if "sr" not in self._do_state:
            raise RuntimeError("set/reset line not configured (sr_mode='off')")
        if kind not in ("set", "reset"):
            raise ValueError(f"kind must be 'set' or 'reset', not {kind!r}")
        end = (kind == "set") == self.cfg.sr_set_on_rising   # line level after the wanted edge
        levels = (not end, end) if self._do_state["sr"] == end else (end,)
        for level in levels:
            self._do_state["sr"] = level
            self._write_do()
            time.sleep(self.cfg.sr_hold)

    def close(self):
        """Reset the relay line to normal polarity and release both tasks.

        The S/R line keeps its level: changing it would fire one more pulse.
        """
        try:
            if "pol" in self._do_state:
                self._do_state["pol"] = False
                self._write_do()
        finally:
            try:
                self.ai.close()
            finally:
                if self.do is not None:
                    self.do.close()


# =====================================================================================
#  Simulated instruments (for --simulate dry runs without hardware)
# =====================================================================================
# The Sim* classes have the same methods as the real ones, so run() does not need to know
# which it uses. They all share one _SimWorld that holds the "physical" state (current,
# relay position, output on/off) and computes a fake sensor signal from it.
# The fake sensor has NO hysteresis: simulated runs test the program flow, not the analysis.
class _SimWorld:
    def __init__(self, cfg: Config):
        self.cfg, self.I, self.sign, self.on = cfg, 0.0, 1, False
        self.sr = 1        # sensor magnetisation: +1 after a set pulse, -1 after a reset pulse
        self.R_coil = 4.0  # ohm (only used in voltage mode: I = V / R)

    def B(self) -> float:  # tesla, incl. a bit of earth-field offset
        return self.sign * self.I * self.on * self.cfg.coil_constant_T_per_A() + 20e-6

    def amr(self) -> float:  # HMC1021S-like bridge at 5 V, 1.0 mV/V/G, linear to ~+/-6 G
        g = self.B() * 1e4    # tesla -> gauss
        # A reset pulse flips the sign of the sensitivity; the 0.5 mV bridge offset stays.
        # tanh gives 1 mV/V/G at small fields and a soft saturation beyond the +/-6 G range.
        return self.sr * 5 * 1.0e-3 * 8.0 * math.tanh(g / 8.0) + 0.5e-3   # volts


class SimPowerSupply(PowerSupply):
    def __init__(self, cfg, world):
        # Deliberately does not call PowerSupply.__init__ (that would open a serial port).
        self.cfg, self.w, self.level = cfg, world, 0.0

    def idn(self): return "SIMULATED EA-PS 2042-10B"
    def setup(self): self.level = 0.0
    def output(self, on): self.w.on = on

    def _set(self, x):
        self.level = x
        self.w.I = x if self.cfg.sweep_mode == "current" else x / self.w.R_coil

    def ramp_to(self, target, dt=0.1):
        super().ramp_to(target, dt=0.0)   # same stepping logic, but no waiting

    def read(self):
        # Round the current to steps of 1/2560 A to imitate the PSU's limited resolution.
        I = round(self.w.I * self.w.on * 2560) / 2560
        return {"V": I * self.w.R_coil, "I": I, "mode": "CC" if self.cfg.sweep_mode == "current" else "CV",
                "OVP": False, "OCP": False, "OPP": False, "OTP": False}

    def close(self): self.ramp_to(0.0); self.output(False)


class SimKeithley(Keithley):
    def __init__(self, cfg, world): self.cfg, self.w = cfg, world
    def idn(self): return "SIMULATED KEITHLEY 2000"
    def setup(self): pass
    def read(self): return self.w.amr() + random.gauss(0, 2e-6), 2e-6   # 2 uV noise
    def close(self): pass


class SimDAQ(MyDAQ):
    def __init__(self, cfg, world): self.cfg, self.w = cfg, world
    def idn(self): return "SIMULATED myDAQ"

    def read(self):
        # The shunt channel returns I * R (+ 50 uV noise); every other channel the
        # (amplified x100) sensor signal plus 1 mV noise.
        out = []
        for lab in self.cfg.daq_labels:
            if self.cfg.is_shunt(lab):
                i = self.w.I * self.w.on * (self.w.sign if self.cfg.shunt_position == "coil" else 1)
                out.append((i * self.cfg.shunt_ohm + random.gauss(0, 5e-5), 2e-4))
            else:
                out.append((self.w.amr() * 100 + random.gauss(0, 1e-3), 3e-3))
        return out

    def set_polarity(self, negative): self.w.sign = -1 if negative else 1
    def sr_pulse(self, kind): self.w.sr = 1 if kind == "set" else -1
    def close(self): pass


# =====================================================================================
#  The connected instruments
# =====================================================================================
class Rig:
    """The instruments of one run, the simulated world (if any) and the coil polarity.

    connect() opens the instruments one by one and stores each as soon as it is open, so a
    failure halfway still leaves everything already opened for close() to shut down. Use it as

        rig = Rig(cfg, simulate)
        try:
            rig.connect(...)
            ...
        finally:
            rig.close()
    """

    def __init__(self, cfg: Config, simulate: bool):
        self.cfg, self.simulate = cfg, simulate
        self.world = _SimWorld(cfg) if simulate else None
        self.psu: PowerSupply | None = None
        self.keith: Keithley | None = None
        self.daq: MyDAQ | None = None
        self.polarity = 1                 # present coil polarity: +1 normal, -1 reversed

    @property
    def relay(self) -> bool:
        """True if a bipolar sweep reverses the coil with the myDAQ-driven relay."""
        return self.cfg.bipolar and self.cfg.polarity_switch == "relay"

    def connect(self, psu: bool = True) -> dict:
        """Open the instruments; return {name: identification string} for the metadata."""
        c, sim, w = self.cfg, self.simulate, self.world
        if psu:
            self.psu = SimPowerSupply(c, w) if sim else PowerSupply(c)
        if c.use_keithley:
            self.keith = SimKeithley(c, w) if sim else Keithley(c)
        if c.needs_daq():
            self.daq = SimDAQ(c, w) if sim else MyDAQ(c)
        ids = {}
        for name, inst in (("psu", self.psu), ("keithley", self.keith), ("daq", self.daq)):
            if inst is not None:
                ids[name] = inst.idn()
                print(f"  {name:9}: {ids[name]}")
        return ids

    def close(self):
        """Close everything, PSU first (ramp down + output off).

        A failure closing one instrument is printed but does not stop the others from closing.
        """
        for inst in (self.psu, self.keith, self.daq):
            if inst is not None:
                try:
                    inst.close()
                except Exception as e:
                    print(f"  close failed for {type(inst).__name__}: {e!r}")


# =====================================================================================
#  Sweep logic
# =====================================================================================
def build_setpoints(cfg: Config) -> list[float]:
    """Setpoints of ONE loop, in the order they are visited.

    Unipolar: 0 -> +max -> 0, e.g. max 2, step 0.05 gives 0, 0.05, ..., 2.0, 1.95, ..., 0
              (41 up + 40 down = 81 points). The maximum is visited once.
    Bipolar : continues 0 -> -max -> 0 (another 80 points, 161 in total). Negative values
              mean "coil reversed"; the polarity is switched when the sign changes.
    round(..., 6) removes floating-point noise such as 0.15000000000000002.
    """
    n = round(cfg.sweep_max / cfg.sweep_step)
    up = [round(i * cfg.sweep_step, 6) for i in range(n + 1)]
    if up[-1] < cfg.sweep_max:
        up.append(cfg.sweep_max)      # max not a multiple of the step: add it explicitly
    branch = up + up[-2::-1]                       # 0 -> +max -> 0
    if cfg.bipolar:
        branch += [-x for x in up[1:]] + [-x for x in up[-2:0:-1]] + [0.0]  # -> -max -> 0
    return branch


def list_instruments():
    """Print every serial port, VISA resource and NI-DAQmx device this PC can see.

    Each section has its own try/except, so a missing driver only affects its own section.
    """
    print("== Serial (COM) ports ==")
    try:
        from serial.tools import list_ports
        for p in list_ports.comports():
            print(f"  {p.device:8} {p.description}")
    except Exception as e:
        print("  ", e)
    print("== VISA resources ==")
    try:
        import pyvisa
        for r in pyvisa.ResourceManager().list_resources():
            print("  ", r)
    except Exception as e:
        print("  ", e)
    print("== NI-DAQmx devices ==")
    try:
        import nidaqmx.system
        for d in nidaqmx.system.System.local().devices:
            print(f"   {d.name}  ({d.product_type})")
    except Exception as e:
        print("  ", e)


# Fixed CSV columns (before the sensor channels). SHUNT_COLUMNS only when a shunt is read.
SWEEP_COLUMNS = ("t_s", "loop", "point", "setpoint", "polarity", "psu_V", "psu_I", "psu_mode",
                 "I_coil_A", "B_mT", "B_set_mT")
SHUNT_COLUMNS = ("I_shunt_A", "B_shunt_mT")


def sensor_labels(cfg: Config, rig: Rig) -> list[str]:
    """Names of the channels read at every point: Keithley first, then the myDAQ inputs."""
    return ([cfg.keithley_label] if rig.keith else []) + (list(cfg.daq_labels) if cfg.use_daq else [])


def csv_header(cfg: Config, rig: Rig) -> list[str]:
    """CSV column names.

    Fixed sweep columns, then I_shunt_A / B_shunt_mT (the measured coil current and its field:
    the standard), then a value + std column per channel. I_coil_A / B_mT are the PSU read-back.
    In flip mode <label> is the reading after the set pulse, <label>_reset after the reset pulse,
    and <label>_sr = (set - reset)/2 the offset-free signal. Its _std column is
    sqrt(std_set^2 + std_reset^2)/2, so std/sqrt(N) is still the error of the mean. The shunt
    channel is not a sensor: in flip mode it gets no _reset/_sr columns; its set and reset
    readings are averaged instead.
    The column names are kept stable so analyse_hysteresis.py and older files stay compatible.
    """
    header = list(SWEEP_COLUMNS)
    if cfg.has_shunt():
        header += list(SHUNT_COLUMNS)
    for lab in sensor_labels(cfg, rig):
        header += [lab, lab + "_std"]
        if cfg.sr_mode == "flip" and not cfg.is_shunt(lab):
            header += [lab + "_reset", lab + "_reset_std", lab + "_sr", lab + "_sr_std"]
    return header


def read_sensors(cfg: Config, keith, daq) -> list[tuple[str, float, float]]:
    """(label, mean, std) in volts for every enabled channel: Keithley first, then myDAQ."""
    vals = []
    if keith:
        vals.append((cfg.keithley_label, *keith.read()))
    if cfg.use_daq:
        vals += [(lab, v, s) for lab, (v, s) in zip(cfg.daq_labels, daq.read())]
    return vals


def measure_sensors(cfg: Config, rig: Rig) -> tuple[list[str], float | None, list[str]]:
    """Read every channel at the present point, with the set/reset pulses sr_mode asks for.

    Returns (CSV cells in csv_header order, shunt voltage or None, console message parts).
    """
    if cfg.sr_mode in ("point", "flip"):
        rig.daq.sr_pulse("set")
    got = read_sensors(cfg, rig.keith, rig.daq)
    got_r = None
    if cfg.sr_mode == "flip":
        rig.daq.sr_pulse("reset")
        got_r = read_sensors(cfg, rig.keith, rig.daq)

    cells, v_shunt, msg = [], None, []
    for n, (lab, v, s) in enumerate(got):
        if got_r is None:                 # one reading per channel
            cells += [f"{v:.9g}", f"{s:.3g}"]
            if cfg.is_shunt(lab):
                v_shunt = v
            else:
                msg.append(f"{lab} {v:+.6f} V")
            continue
        _, vr, sr = got_r[n]              # flip: set and reset reading per channel
        s_comb = math.hypot(s, sr) / 2
        if cfg.is_shunt(lab):             # the current does not flip: average both readings
            v_shunt = (v + vr) / 2
            cells += [f"{v_shunt:.9g}", f"{s_comb:.3g}"]
        else:
            d = (v - vr) / 2
            cells += [f"{v:.9g}", f"{s:.3g}", f"{vr:.9g}", f"{sr:.3g}", f"{d:.9g}", f"{s_comb:.3g}"]
            msg.append(f"{lab} (S-R)/2 {d * 1e3:+.3f} mV")
    return cells, v_shunt, msg


def measured_current(cfg: Config, rig: Rig, psu_I: float | None = None) -> float:
    """Signed coil current (A) from the standard source.

    "shunt": a fresh myDAQ reading over the shunt. "psu": the PSU read-back (psu_I if given,
    otherwise read now) times the polarity.
    """
    if cfg.current_source == "shunt":
        v = dict((lab, v) for lab, (v, _) in zip(cfg.daq_labels, rig.daq.read()))[cfg.shunt_label]
        return cfg.shunt_current(v, rig.polarity)
    if psu_I is None:
        psu_I = rig.psu.read()["I"]
    return rig.polarity * psu_I


def wait_for_manual_swap(cfg: Config, rig: Rig, want_pol: int) -> None:
    """Pause the sweep until the user has swapped the coil leads by hand.

    Called with the PSU output already OFF. Checks the coil current is ~0 before asking (with
    the shunt when available: the PSU read-back shows 0 below ~0.05 A anyway), so the leads are
    never pulled while current flows (an inductor would arc). Only a typed 'y' continues;
    Ctrl+C aborts the run (the PSU is already off). In a simulation it just flips the simulated
    polarity and goes on.
    """
    which = "REVERSED" if want_pol < 0 else "NORMAL"
    src = "shunt" if cfg.current_source == "shunt" else "PSU read-back"
    I = measured_current(cfg, rig)
    if abs(I) > 0.05:
        raise RuntimeError(f"coil current still {I:+.3f} A ({src}) with the output off - "
                           f"not asking for a lead swap")
    print(f"\n  >>> PSU output is OFF (I = {I:+.4f} A, {src}).")
    print(f"  >>> Swap the coil leads so the coil is connected {which}.")
    if rig.world is not None:
        rig.world.sign = want_pol
        print("  >>> (simulated - continuing)\n")
        return
    while input(f"  >>> Type 'y' + Enter once the coil is {which} (Ctrl+C aborts): ").strip().lower() != "y":
        pass
    print()


def set_polarity(cfg: Config, rig: Rig, want_pol: int) -> None:
    """Reverse the coil (relay or by hand) if `want_pol` differs from the present polarity.

    Never break the coil circuit while current flows through an inductor: contacts would arc.
    Current to 0, output off, then switch, then output on again.
    """
    if want_pol == rig.polarity:
        return
    rig.psu.ramp_to(0.0)
    rig.psu.output(False)
    time.sleep(0.3)                       # let the current decay fully
    if rig.relay:
        rig.daq.set_polarity(want_pol < 0)
        time.sleep(0.2)                   # let the relay contacts settle
    else:
        wait_for_manual_swap(cfg, rig, want_pol)
    rig.psu.output(True)
    rig.polarity = want_pol


def measure_point(cfg: Config, rig: Rig, loop: int, i: int, sp: float, t0: float) -> tuple[list, str]:
    """Go to setpoint `sp`, measure everything, return (CSV row, console line)."""
    # ---- 1. polarity ------------------------------------------------------------------
    # A setpoint of 0 keeps the present polarity, so the coil is only reversed when the sign
    # really changes.
    set_polarity(cfg, rig, -1 if sp < 0 else (1 if sp > 0 else rig.polarity))

    # ---- 2. ramp and settle -----------------------------------------------------------
    # The PSU only knows magnitudes; the sign is handled by the polarity switch.
    rig.psu.ramp_to(abs(sp))
    time.sleep(cfg.settle_time)

    # ---- 3. power supply read-back and checks -----------------------------------------
    p = rig.psu.read()
    if p["OVP"] or p["OCP"] or p["OPP"] or p["OTP"]:
        raise RuntimeError(f"PSU protection tripped: {p}")
    if cfg.sweep_mode == "current" and abs(sp) > 0 and p["mode"] != "CC":
        # In CV mode the PSU hit its voltage limit and delivers LESS current than requested.
        # The data is still recorded (with the measured current).
        print(f"  ! PSU in CV at {sp} A: raise psu_voltage_limit (coil R too high?)")
    I_psu = rig.polarity * p["I"]
    B_set = cfg.field_mT(sp) if cfg.sweep_mode == "current" else float("nan")  # unknown in V mode

    # ---- 4. sensor readings (incl. the shunt) -----------------------------------------
    cells, v_shunt, sens_msg = measure_sensors(cfg, rig)

    # ---- 5. coil current and field ----------------------------------------------------
    row = [f"{time.perf_counter() - t0:.3f}", loop, i, sp, rig.polarity,
           f"{p['V']:.4f}", f"{p['I']:.5f}", p["mode"],
           f"{I_psu:.5f}", f"{cfg.field_mT(I_psu):.6f}", f"{B_set:.6f}"]
    if cfg.has_shunt():
        I_sh = cfg.shunt_current(v_shunt, rig.polarity)
        row += [f"{I_sh:.6f}", f"{cfg.field_mT(I_sh):.6f}"]
        # A large mismatch means wrong wiring, shunt value or shunt_position.
        if cfg.sweep_mode == "current" and abs(sp) >= 0.1 and abs(I_sh - sp) > 0.1 * abs(sp) + 0.02:
            print(f"  ! shunt reads {I_sh:+.4f} A at setpoint {sp:+.3f} A: check "
                  f"the AI wiring, shunt_ohm ({cfg.shunt_ohm}) and shunt_position")
    row += cells

    # Console: the standard current and its field first, the sensors, then the PSU read-back.
    I_std = I_sh if cfg.current_source == "shunt" else I_psu
    unit = "A" if cfg.sweep_mode == "current" else "V"
    parts = [f"[{loop}:{i:4d}] set {sp:+7.3f} {unit}",
             f"I {I_std:+.4f} A ({cfg.current_source})", f"B {cfg.field_mT(I_std):+8.4f} mT",
             *sens_msg]
    if cfg.current_source == "shunt":
        parts.append(f"I_psu {I_psu:+.4f} A")
    return row, " | ".join(parts)


def run(cfg: Config, simulate: bool, plot: bool) -> Path:
    """Do one complete measurement and return the path of the CSV file.

    Structure:
      try:     connect instruments -> set up -> sweep all loops, writing each point at once
      except:  record why the run stopped (Ctrl+C or error)
      finally: ALWAYS close the instruments (PSU first: current to 0, output off) and
               write the JSON metadata, whatever happened
    """
    cfg.validate()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # File names: <sample_name>_<date>_<time>, e.g. AMR_20261114_135305.csv / .json / .png
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out_dir / f"{cfg.sample_name}_{stamp}"
    csv_path, meta_path = base.with_suffix(".csv"), base.with_suffix(".json")

    k_mT = cfg.field_mT(1.0)
    setpoints = build_setpoints(cfg)
    unit = "A" if cfg.sweep_mode == "current" else "V"
    field_col = "B_shunt_mT" if cfg.current_source == "shunt" else "B_mT"
    print(f"Coil constant: {k_mT:.4f} mT/A | {len(setpoints)} points x {cfg.loops} loop(s) "
          f"| max {cfg.sweep_max} {unit} | coil current from: {cfg.current_source}")

    meta = {"config": asdict(cfg), "coil_constant_mT_per_A": k_mT, "simulated": simulate,
            "current_source": cfg.current_source, "field_column": field_col,
            "start": datetime.now().isoformat(timespec="seconds"), "instruments": {}}
    status = "completed"
    rig = Rig(cfg, simulate)
    try:
        # ---- connect and set up ----
        meta["instruments"] = rig.connect()
        rig.psu.setup()
        if rig.keith:
            rig.keith.setup()
        if rig.relay:
            rig.daq.set_polarity(False)   # start with normal polarity
        if cfg.sr_mode != "off":
            rig.daq.sr_pulse("set")       # start from a freshly set sensor (coil still off)

        # ---- sweep ----
        with open(csv_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(csv_header(cfg, rig))
            rig.psu.output(True)          # output on at level 0 (set in setup)
            t0 = time.perf_counter()      # time reference for the t_s column
            for loop in range(cfg.loops):
                for i, sp in enumerate(setpoints):
                    row, msg = measure_point(cfg, rig, loop, i, sp, t0)
                    # flush() pushes the row to disk now, so a crash keeps all points so far.
                    wr.writerow(row)
                    f.flush()
                    print(msg)

    except KeyboardInterrupt:
        # Ctrl+C: not an error, the data so far is valid. Falls through to `finally`.
        status = "aborted by user"
        print("\nInterrupted - shutting down safely...")
    except Exception as e:
        status = f"error: {e!r}"
        print(f"\nERROR: {e!r} - shutting down safely...")
        raise   # re-raise after `finally` has run, so the full error is shown
    finally:
        rig.close()
        if cfg.bipolar and not rig.relay and rig.polarity < 0:
            print("  NOTE: the coil leads are still REVERSED - swap them back before the next run.")
        meta["end"] = datetime.now().isoformat(timespec="seconds")
        meta["status"] = status
        meta["csv"] = csv_path.name
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Status: {status}\nData: {csv_path.resolve()}\nMeta: {meta_path.resolve()}")

    if plot and csv_path.exists():
        make_plot(csv_path, cfg.shunt_label if cfg.has_shunt() else "")
    return csv_path


def sr_test(cfg: Config, simulate: bool, cycles: int) -> int:
    """Check the set/reset pulses WITHOUT coil current (the PSU is not touched).

    Fires `cycles` set/reset pairs and prints every channel after each pulse. If the pulses
    reach the strap, the bridge output jumps between two levels whose half-difference
    (S-R)/2 is the sensor's response to the field already there (Earth's field: a few mV at
    5 V supply, depending on orientation; rotating the board changes it). The supply channel
    should not change. (S-R)/2 = 0 within the noise means no pulses reach the strap, or the
    sensor does not respond.
    """
    cfg.bipolar = False                     # no relay needed here
    cfg.use_daq = True                      # the test reads the myDAQ (and the Keithley if enabled)
    if cfg.sr_mode == "off":
        cfg.sr_mode = "flip"                # any value except "off" sets up the S/R line
    rig = Rig(cfg, simulate)
    try:
        rig.connect(psu=False)
        if rig.keith:
            rig.keith.setup()
        print(f"S/R test on {cfg.daq_device}/{cfg.sr_line} ({'rising' if cfg.sr_set_on_rising else 'falling'}"
              f" edge = set), {cycles} cycles, no coil current")
        diffs: dict[str, list[float]] = {}
        for n in range(cycles):
            rig.daq.sr_pulse("set")
            a = read_sensors(cfg, rig.keith, rig.daq)
            rig.daq.sr_pulse("reset")
            b = read_sensors(cfg, rig.keith, rig.daq)
            for (lab, vs, _), (_, vr, _) in zip(a, b):
                d = (vs - vr) / 2
                diffs.setdefault(lab, []).append(d)
                print(f"[{n + 1:3d}] {lab:18} set {vs:+.6f} V  reset {vr:+.6f} V  (S-R)/2 {d * 1e3:+8.3f} mV")
        print("Mean (S-R)/2 per channel:")
        for lab, d in diffs.items():
            sem = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
            print(f"  {lab:18} {statistics.fmean(d) * 1e3:+8.3f} mV  +/- {sem * 1e3:.3f} mV")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        rig.close()
    return 0


def make_plot(csv_path: Path, shunt_label: str = ""):
    """Quick-look plot: every signal column against B, one panel per column, saved as PNG.

    This is only for checking a run right after measuring; analyse_hysteresis.py does the
    real analysis. The shunt voltage column (shunt_label) is the x axis, not a panel.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")         # render to file only, no window
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipping plot)")
        return
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    # Signal columns = everything except the fixed sweep columns and the *_std columns.
    fixed = {*SWEEP_COLUMNS, *SHUNT_COLUMNS, shunt_label}
    # A *_reset column (sr_mode "flip") is drawn in the same panel as its set reading.
    cols = [c for c in rows[0] if c not in fixed and not c.endswith(("_std", "_reset"))]
    if not cols:
        return
    # Field from the shunt current (the standard) when measured; otherwise from the setpoint
    # when known (the PSU read-back current is 0 below ~0.05 A and scatters by ~10 mA, see
    # analyse_hysteresis.py --field). Voltage mode without a shunt only has B_mT.
    if "B_shunt_mT" in rows[0]:
        bcol, bsrc = "B_shunt_mT", "shunt current"
    elif all(r.get("B_set_mT", "nan") != "nan" for r in rows):
        bcol, bsrc = "B_set_mT", "setpoint"
    else:
        bcol, bsrc = "B_mT", "PSU read-back"
    B = [float(r[bcol]) for r in rows]
    fig, axes = plt.subplots(len(cols), 1, figsize=(7, 3 * len(cols)), sharex=True, squeeze=False)
    for ax, c in zip(axes[:, 0], cols):
        if c + "_reset" in rows[0]:
            ax.plot(B, [float(r[c]) for r in rows], ".-", ms=3, lw=0.8, label="after set")
            ax.plot(B, [float(r[c + "_reset"]) for r in rows], ".-", ms=3, lw=0.8, label="after reset")
            ax.legend(fontsize=8)
        else:
            ax.plot(B, [float(r[c]) for r in rows], ".-", ms=3, lw=0.8)
        ax.set_ylabel(f"{c} (V)")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel(f"B at coil centre (mT, from {bsrc})")
    fig.suptitle(csv_path.stem)
    fig.tight_layout()
    png = csv_path.with_suffix(".png")
    fig.savefig(png, dpi=150)
    print(f"Plot: {png.resolve()}")


# =====================================================================================
def build_parser(cfg: Config) -> argparse.ArgumentParser:
    """Command-line options; their defaults come from `cfg`."""
    ap = argparse.ArgumentParser(description="AMR sensor sweep in a Helmholtz coil (EA-PS 2042-10B + Keithley + myDAQ)")
    ap.add_argument("--list", action="store_true", help="list available instruments and exit")
    ap.add_argument("--simulate", action="store_true", help="use fake instruments (dry run)")
    ap.add_argument("--psu-port", default=cfg.psu_port)
    ap.add_argument("--keithley", default=cfg.keithley_resource, help="VISA resource string")
    ap.add_argument("--keithley-type", choices=["dmm", "smu"], default=cfg.keithley_type)
    ap.add_argument("--daq", default=cfg.daq_device, help="myDAQ device name (NI MAX)")
    ap.add_argument("--mode", choices=["current", "voltage"], default=cfg.sweep_mode)
    ap.add_argument("--max", type=float, default=cfg.sweep_max, help="max coil current (A) / voltage (V)")
    ap.add_argument("--step", type=float, default=cfg.sweep_step)
    ap.add_argument("--loops", type=int, default=cfg.loops)
    ap.add_argument("--settle", type=float, default=cfg.settle_time, help="settle time per point (s)")
    ap.add_argument("--bipolar", action="store_true", help="sweep negative too (see --polarity)")
    ap.add_argument("--polarity", choices=["manual", "relay"], default=cfg.polarity_switch,
                    help="bipolar: 'manual' pauses at 0 A for you to swap the coil leads, "
                         "'relay' switches a relay from a myDAQ DO line")
    ap.add_argument("--turns", type=int, default=cfg.coil_turns, help="turns per coil")
    ap.add_argument("--radius", type=float, default=cfg.coil_radius_m, help="coil radius (m)")
    ap.add_argument("--current-source", choices=["shunt", "psu"], default=None,
                    help=f"coil current for the field: 'shunt' = myDAQ over the shunt on "
                         f"{cfg.shunt_label} (default), 'psu' = PSU read-back (default with --no-daq)")
    ap.add_argument("--shunt", type=float, default=cfg.shunt_ohm,
                    help=f"coil-current shunt resistance (ohm), read on {cfg.shunt_label}")
    ap.add_argument("--sr", choices=["off", "once", "point", "flip"], default=cfg.sr_mode,
                    help="HMC1021S set/reset pulses: once (before the sweep), point (set before every "
                         "point) or flip (set+reset at every point, adds offset-free <label>_sr columns)")
    ap.add_argument("--sr-line", default=cfg.sr_line, help="myDAQ DO line feeding the S/R pulse driver")
    ap.add_argument("--sr-test", type=int, metavar="N",
                    help="no coil current: fire N set/reset pairs, print the readings, exit")
    ap.add_argument("--no-keithley", action="store_true")
    ap.add_argument("--no-daq", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--name", default=cfg.sample_name, help="sample name used in file names")
    ap.add_argument("--out", default=cfg.out_dir, help="output folder")
    ap.add_argument("--notes", default=cfg.notes, help="free text stored in the metadata")
    return ap


def apply_args(cfg: Config, a: argparse.Namespace) -> None:
    """Copy the command-line options into the Config."""
    cfg.psu_port, cfg.keithley_resource, cfg.keithley_type = a.psu_port, a.keithley, a.keithley_type
    cfg.daq_device, cfg.sweep_mode, cfg.sweep_max, cfg.sweep_step = a.daq, a.mode, a.max, a.step
    cfg.loops, cfg.settle_time, cfg.bipolar = a.loops, a.settle, a.bipolar
    cfg.polarity_switch = a.polarity
    cfg.coil_turns, cfg.coil_radius_m, cfg.shunt_ohm = a.turns, a.radius, a.shunt
    # From the command line both instruments are used unless --no-keithley / --no-daq is given.
    cfg.use_keithley, cfg.use_daq = not a.no_keithley, not a.no_daq
    cfg.sample_name, cfg.out_dir, cfg.notes = a.name, a.out, a.notes
    cfg.sr_mode, cfg.sr_line = a.sr, a.sr_line
    if a.current_source is not None:
        cfg.current_source = a.current_source
    if a.simulate:
        cfg.settle_time = 0.0             # nothing to settle in a simulation: run fast
        cfg.sr_hold = 0.0


def main(argv=None):
    """Command line: read the options into a Config, ask for confirmation, then run().

    Settings without an option here (protection limits, daq_rate, daq_samples, NPLC,
    ramp_rate, ...) are changed by editing the Config defaults above.
    """
    cfg = Config()
    a = build_parser(cfg).parse_args(argv)
    if a.list:
        list_instruments()
        return 0
    apply_args(cfg, a)

    if a.sr_test:
        return sr_test(cfg, a.simulate, a.sr_test)

    # The shunt is the standard; without the myDAQ inputs it cannot be read, so fall back to
    # the PSU read-back unless --current-source shunt was asked for explicitly (validate()
    # then refuses the combination).
    if a.current_source is None and not cfg.has_shunt():
        cfg.current_source = "psu"
        print("NOTE: no shunt reading (--no-daq or no shunt channel): "
              "the coil current comes from the PSU read-back.")
    try:
        cfg.validate()                    # fail before the confirmation prompt, not after
    except ValueError as e:
        print(f"Invalid settings: {e}")
        return 2

    if not a.simulate:
        # Last chance to stop before real current flows through the coil.
        print("WARNING: the PSU output will be switched ON and drive the coil.")
        if cfg.bipolar and cfg.polarity_switch == "manual":
            print("Bipolar, manual polarity: connect the coil with NORMAL polarity now. The sweep")
            print("will pause at 0 A with the output OFF each time the leads must be swapped.")
        if input("Type 'y' to start: ").strip().lower() != "y":
            print("Aborted.")
            return 1
    run(cfg, simulate=a.simulate, plot=not a.no_plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
