# AMR-sensor
Repository containing code, for driving a helmoltz-coil, and measuring/characterizing an AMR-sensor.

Installation of drivers
-----------------------
pip install eaps2000 pyserial pyvisa nidaqmx numpy matplotlib pyvisa-py


AMR sensor characterisation in a Helmholtz coil.
------------------------------------------------
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
