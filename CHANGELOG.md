# Changelog

Notable changes to R&S NGM FastLog. Versions follow [Semantic Versioning](https://semver.org/).

## [1.1.0] — 2026-11-01

### Added

- **Markers.** X1 and X2 mark times across all plots; Y1 and Y2 mark levels in one plot
  (voltage, current or power). Each is a labelled line you drag to move, in magenta for X and
  green for Y. Place them with the **Markers** button, or at a point from a plot's right-click
  menu. The Measurements panel shows the time at X1 and X2 with each channel's V, I and P
  there, ΔX = X2 − X1 and 1/ΔX; and the levels at Y1 and Y2, ΔY = Y2 − Y1 and 1/ΔY.
  **Clear** removes them.
- **Selection box.** With **Select** on, dragging in a plot draws a translucent box instead
  of panning. The Measurements panel shows, for each channel, the points in the box's time
  range, the points in the box (also between its levels), the time in the box (with its share
  of the time range, e.g. the time above a threshold), and the min, mean, max and σ of the
  points in the box. Drag a corner to resize the box, or with Select on the inside to move it;
  **Esc** clears it.
- **Measurements panel.** Shown beside the plots while there are markers or a selection.
  **Pop Out** moves it into a window of its own (closing it clears the markers and the
  selection), and **Dock** puts it back. Docked or popped out is remembered.

### Fixed

- **Plots resizing as the mouse moved.** The cursor readout under the plots changed width
  with the values under the mouse, which changed where the statistics beside it wrapped. The
  plots then grew and shrank as the statistics took more or fewer lines. Most noticeable with
  the Measurements panel open. The readout now keeps the width of its longest text, and the
  statistics keep the height their longest text needs at the current window width, so the
  plots resize only with the window.

### Also

- Tests drive the markers and the selection box with the mouse, and check that the plots
  keep their size as the readouts change (66 tests in total).

## [1.0.0] — 2026-11-01

First stable release. The app is the same as 0.2.0, which passed preliminary testing on an
NGM202 with a Bluetooth device on CH1. The known limitations listed under 0.1.0 still apply.

## [0.2.0] — 2026-10-19

### Changed

- **Output switch.** Each channel's output is a two-position OFF / ON switch instead of a
  button labelled with its state, so it's clear that clicking it switches the output.
- **Connecting only reads.** Channel settings (measurement ranges) are no longer remembered
  between sessions. The channel panels show "—" until the app connects, then show the
  instrument's setpoints, output states and ranges, and keep following changes made on the
  front panel. Ranges are applied when you choose them.

### Fixed

- **Stale setpoint sent.** A setpoint field that had keyboard focus when the instrument's
  values arrived kept its old value (e.g. 0.000 V), and leaving the field then sent that value
  to the instrument. Fields now send only values you typed or stepped.
- **Accidental setpoint changes.** The mouse wheel no longer changes a setpoint field you
  scroll past; click into the field first.
- **Plots out of line with the time axis.** Each plot's y axis took the width of its widest
  tick label, so a plot whose labels had more digits (or gained decimals when zooming) had
  its plot area shifted sideways, and the times on it no longer lined up with the time axis
  under the bottom plot. The y axes now have a fixed width with room for 8 significant digits,
  so the plots always line up and don't move when zooming. Tick labels never show more digits
  than fit.

### Also

- Tests for the main window against the simulator check that connecting sends no setting
  commands, and that the plot areas stay aligned (40 tests in total).

## [0.1.0] — 2026-10-18

First version, for review: a desktop GUI and Python driver for FastLog voltage and current
logging with Rohde & Schwarz NGM201 and NGM202 power supplies over LAN.

### Instrument control

- **Connection.** Connect by IP address, hostname, or VISA resource string over a raw socket
  (no VISA installation needed), HiSLIP or VXI-11, or find instruments with **Scan**. Uses
  Rohde & Schwarz's RsInstrument library.
- **Channels.** Per channel: voltage setpoint, current limit, output switch, live readback
  (V, I, P, CV/CC, protection trips) and measurement ranges (5/20 V; 10 mA to 10 A; or
  Auto). Setpoints and outputs stay live during captures.
- **Restart.** Reboot the instrument from the app and reconnect automatically, to recover
  FastLog after an internal error.

### Acquisition

- **FastLog streaming** from CH1, CH2 or both, at 100 Sa/s to 500 kSa/s per channel (500 kSa/s
  combined). Two channels start on one trigger, so their samples are time-aligned.
- **Trigger modes:** Single, Auto (seamless back-to-back captures) and Manual (Trigger button).
- **Output switching:** optionally switch the measured outputs on once FastLog is running,
  to capture the turn-on and inrush, and off at the end.
- **Lost data detection.** Chunks the instrument replaced before they were read are detected
  from the chunk timing and stored as NaN gaps at the right place in time.
- **Clipped readings.** Readings beyond the measurement range are flagged, for fixed ranges
  and autorange.
- **Safeguards.** The app never configures a rate combination that locks FastLog up, stops any
  FastLog left running before a run, and refuses to start if other instrument functions are
  set to react to the trigger.

### Plotting

- Stacked voltage, current and power plots on a shared time axis, CH1 orange and CH2 blue.
- Long captures stay responsive: redraws use min/max summaries, about 20 ms for 15 million
  samples.
- Zoom buttons, mouse wheel and Box Zoom; Line, Line + points or Points styles.
- Red bands for lost samples and amber bands for clipped readings, with **◀ ▶** navigation and
  a warning bar.
- Statistics for the displayed window: mean, σ, min, max of V, I and P, energy and charge.

### Saving

- **Save View** (displayed window) and **Save All**, as CSV (time, then voltage, current and
  flags per channel) or HDF5 (lossless float32 per channel, plus gap and clipped runs), with
  instrument, setpoint and range metadata. Long saves show progress and can be cancelled.

### Also included

- **Driver.** `ngm_daq.instrument` and `ngm_daq.capture` can be used without the GUI.
- **Simulator.** `tools/ngm_simulator.py` emulates the instrument, including its FastLog timing
  and quirks.
- **Tests.** 28 pytest tests: data model, export, and the worker against the simulator.
- **Pinned dependencies.** `uv.lock` pins exact versions; Python 3.14 is pinned in
  `.python-version`, and 3.12 or later is supported.

### Verified

On an NGM202 (firmware 04.013, options K103, K104, K106) with a Bluetooth device on CH1, on
Fedora 43:

- CH1 at 500 kSa/s for 30 s (15 M samples) with no lost samples while the plot was zoomed and
  panned; 3 s captures with the turn-on and inrush.
- Both channels at 250 kSa/s; Auto mode with 8 seamless back-to-back captures at 500 kSa/s.
- Clipping on the 10 mA and 100 mA ranges flagged where the trace is pinned.

### Known limitations

- The NGM201, other firmware versions, Windows and macOS are untested.
- A capture is limited by memory: about 8 minutes of one channel at 500 kSa/s per 8 GB free.
- Only immediate and software (Trigger button) starts; external triggers are planned.
