Detector and Goniometer Calibration
====================================

Covers the **Peak/Shape/Refinement** and **Calibration** tabs, which
together turn a run (or set of runs) on a well-characterized standard
crystal into an updated detector geometry and goniometer model. The
workflow applies to any of the supported instruments (TOPAZ, MANDI,
CORELLI, SNAP).

The physical idea
------------------
Every panel on the detector array, and the sample position itself, is
installed by hand and only *approximately* matches the idealized
geometry baked into the instrument definition. Indexing Bragg peaks
against that idealized geometry produces d-spacings that disagree
slightly from the true, tabulated d-spacings of the standard - the
size and pattern of that disagreement shows exactly how each panel is
out of place.

A calibration standard is chosen specifically because its lattice
spacings are known essentially exactly (the GUI defaults to a silicon
sphere, a=5.431 Å cubic, but anything with a well-established
structure works). That turns the crystal itself into a ruler: any
residual between the predicted and observed peak position is
attributed to the instrument geometry, not to uncertainty in the
sample. More strong peaks, spread across as many panels and as wide a
d-spacing range as possible, better constrain the geometry fit.

The same run set also pins down the goniometer: once the detector
geometry is trusted, comparing where a peak *should* land (from the
motor angles logged for that run) against where it actually landed
across many different rotation settings reveals whether the physical
rotation stack is tilted relative to the axes the software assumes,
and whether a fixed offset is baked into one of the encoders.

Step 1: Find strong peaks (Peak/Shape/Refinement tab)
-------------------------------------------------------
- Switch to the **Peak/Shape/Refinement** tab.
- Set **Instrument** and **IPTS**, and enter the **Runs** collected on
  the standard (e.g. ``12345:12360``). Several runs at different
  goniometer settings give the fit angular leverage; a single run only
  constrains the panels seen at that one orientation.
- Enter the standard's **unit cell** (lengths/angles), **Crystal
  System**, **Lattice System**, and **Centering** - the known ruler,
  not a value to be refined here.
- **Max Threshold** and **Peak Radius** control how aggressively peaks
  are searched for and how large a region around each peak center is
  integrated; the defaults are reasonable starting points. Each run is
  converted to reciprocal space and scanned for tight, locally dense
  clusters of scattered neutrons - real Bragg peaks - while
  automatically dialing the density threshold per run so that a run is
  neither swamped by noise nor starved by a threshold tuned for a
  differently-oriented run.
- An existing **Detector Calibration** / **Tube Calibration** / **UB
  File** can optionally be supplied as a starting point when a prior
  calibration or orientation is already available; otherwise the
  search works from the bare cell parameters alone.
- Click **Run**. For each run, garnet works out the crystal orientation
  from the found peaks and the known cell, indexes every peak to hkl,
  and integrates a small fixed region around each peak center. Runs
  are then merged into one combined peaks table and reoriented onto a
  common setting so peaks collected at different goniometer angles
  line up with each other.
- The output folder contains ``peaks.nxs`` (the peaks table used as
  input to calibration), ``peaks.mat`` (the fitted orientation), and
  ``mdhkl.nxs``, a merged reciprocal-space map for checking that
  coverage is reasonably complete and symmetric.

Step 2: Refine the detector geometry (Calibration tab)
---------------------------------------------------------
- Switch to the **Calibration** tab (inside the **Calibration**
  sub-tab).
- Point **Peaks Table** at the ``peaks.nxs`` produced above (or any
  other already-indexed peaks file), and re-enter the standard's unit
  cell, crystal system, and lattice system.
- Leave **Refine Goniometer** unchecked for now - the first pass is
  purely about panel geometry.
- Click **Run**. Each pass:

  1. Strips the goniometer setting out of every peak first, so peaks
     from every run are pooled onto the crystal's own frame regardless
     of what angle they were collected at. This lets every run
     contribute statistics to every panel it touched.
  2. Adjusts each panel's position and tilt, and the overall
     sample-to-moderator distance, to minimize the mismatch between
     the observed and the true, tabulated d-spacing across every peak
     on every panel simultaneously.
  3. Re-centers the geometry on the fitted sample position, since a
     shift in the sample position is otherwise indistinguishable from
     a shift of every panel at once.
  4. Writes a diagnostic PDF with one plot per detector bank showing
     percent d-spacing residual versus d-spacing, before and after the
     fit, plus an instrument-wide coverage map. A well-calibrated panel
     shows its points collapse onto the zero line; a panel that still
     trends up or down, or is offset from its neighbors, usually needs
     another look (bad peaks, wrong mask, or a genuinely stubborn
     misalignment).

- More **Iterations** repeats the pass when a panel's residual hasn't
  flattened out after one round - each iteration re-derives peak
  positions from the newly calibrated geometry before fitting again.

Step 3: Refine the goniometer (optional)
------------------------------------------
- Once the panel residuals look flat, re-run with **Refine
  Goniometer** checked.
- This step no longer touches panel positions. Instead, it uses the
  peaks in their now-trusted lab-frame positions, together with the
  motor angles logged for each run, to fit a single rigid rotation-axis
  tilt and encoder offset that best explains every peak in every run at
  once. Physically, this captures a goniometer cradle that isn't
  perfectly plumb - a small, fixed tilt of the rotation axis relative
  to what the software assumes, plus a fixed angular offset on one axis
  (e.g. chi) between the encoder reading and the true zero.
- The tilt direction is constrained to be consistent with how a
  horizontal shaft would sag under its own weight, which removes an
  otherwise-ambiguous rotational degree of freedom in the fit (without
  that constraint, an arbitrary combination of tilt and beam-axis
  rotation would fit the data equally well).
- The fitted tilt and offset are written into the instrument
  parameters alongside the panel geometry, and a new diagnostic PDF is
  generated the same way as in Step 2, confirming that the goniometer
  pass did not reintroduce any panel residual.

Reading the output
-------------------
- ``calibration.xml`` / ``calibration.DetCal`` - the refined panel
  geometry, used as the **Detector Calibration** input for the
  **Peak/Shape/Refinement**, **Vanadium**, and main reduction tabs for
  subsequent experiments on the same cycle.
- ``calibration_goniometer.xml`` - the fitted rotation-axis tilt and
  encoder offset, present only if goniometer refinement was run.
- ``calibration_<n>.pdf`` - per-iteration diagnostics; the last
  iteration compared against the first shows how much the fit
  improved.
- ``goniometer.txt`` - the fitted tilt angles and chi offset in plain
  text, useful for tracking calibration drift across cycles.
