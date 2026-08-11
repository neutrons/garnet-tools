Generating Vanadium Normalization Files
=========================================

Covers the **Vanadium** sub-tab (inside the **Calibration** tab),
which turns a vanadium run into the correction files used to normalize
Bragg peak intensities in the main reduction pipeline. The workflow
applies to any of the supported instruments.

The physical idea
------------------
A raw Bragg peak's integrated counts depend on more than the sample's
crystal structure: how many neutrons of that wavelength the source and
guide actually delivered that day, how efficiently the particular
pixels the peak landed on convert a neutron into a recorded count, and
how much solid angle those pixels subtend as seen from the sample.
None of that is physics that belongs in the final structure - it is
instrumental, and it has to be measured and divided out.

Vanadium is used to measure it because vanadium's coherent (Bragg)
scattering is almost zero. A vanadium sphere or rod in the beam
produces an almost purely elastic, incoherent signal with no crystal
structure of its own - a smooth, structureless response whose only
remaining wavelength- and pixel-dependence comes from exactly the
three instrumental effects above. That makes a vanadium run a flat
field: dividing a real sample's data by it (after a few corrections)
leaves the sample's own scattering.

Step 1: Runs and sample geometry
----------------------------------
- Switch to the **Vanadium** sub-tab.
- Set **Instrument**, **Vanadium IPTS/Runs**, and the matching
  **No-Sample IPTS/Runs** - a background run taken under the same beam
  conditions but with the vanadium standard removed. The background is
  subtracted from the vanadium run to strip out scattering from the
  sample environment itself (can walls, air, cryostat) that would
  otherwise contaminate the "pure vanadium" signal.
- Choose the **Sample Shape** (sphere or cylinder) and enter its
  **Diameter** (and **Height** for a cylinder) to match the physical
  standard actually mounted in the beam. This isn't cosmetic - it
  drives the absorption correction below.
- **Beam Diameter** matters when the beam is narrower than the
  standard: only the illuminated volume contributes signal, and a beam
  smaller than the vanadium changes how much of the rod or sphere is
  actually seen.
- **Output Folder** is a short name, not a full path: results are
  written to ``/SNS/<Instrument>/shared/Vanadium/<Output Folder>/``, a
  facility location shared by everyone working on that instrument -
  not a personal or scratch directory. Pick a descriptive name (e.g.
  the cycle or sample geometry).
- ``absorption_parameters.txt`` in that output folder reports the
  linear absorption/scattering coefficients and the equivalent mass
  and density derived from the entered dimensions. Vanadium's density
  is well known (~6.11 g/cm\ :sup:`3`), so a wildly different implied
  density is a quick sign the entered diameter/height doesn't match
  the physical standard.

Step 2: Wavelength range and pixel grouping
----------------------------------------------
- **k(min)/k(max)** set the momentum (equivalently wavelength) window
  the correction covers; this should bracket the full band used for
  the actual sample reduction, and auto-populates from the
  instrument's usable wavelength band when **Instrument** changes.
- **Rows/Cols** sum neighboring pixels together before building the
  correction curves. Vanadium runs are typically much shorter than
  sample runs, so grouping trades a little spatial resolution for a
  correction surface with far less counting-statistics noise - and it
  is applied consistently, so the grouping should match what is used
  for the sample data downstream.
- **Mask Options** exclude dead or noisy tubes/pixels/banks. This
  matters even more here than for a sample run: after grouping, one
  bad pixel can poison the smoothed signal of its whole neighborhood,
  leaving a visible artifact in the final correction map.

Step 3: Run and inspect the correction
-----------------------------------------
- Point **Instrument Definition**, **Detector Calibration**, and
  **Tube Calibration** at the calibrated geometry (see the detector
  calibration guide) - the vanadium correction has to be computed on
  the exact panel geometry used to reduce the sample data, since it
  depends on each pixel's real position and solid angle.
- Click **Run**. Behind the scenes:

  1. Vanadium and background are loaded, normalized by proton charge,
     grouped, and masked identically.
  2. The background is subtracted from the vanadium signal.
  3. An absorption correction is applied: a thicker or larger standard
     absorbs more of the slower (longer-wavelength) neutrons before
     they can scatter back out, especially along the longer path
     lengths seen at large scattering angles. This step estimates, for
     every pixel and wavelength, how much stronger the signal would
     have been without that self-absorption, and scales it back up
     accordingly, using the sample's real size/shape and its known
     absorption coefficient.
  4. The now background-subtracted, absorption-corrected data is split
     into two physically distinct pieces: the incident spectrum (how
     much flux the source delivered at each wavelength, summed over
     the whole array) and a per-bank detector efficiency curve (how
     sensitive each bank is as a function of wavelength, cross
     calibrated bank to bank). Both are smoothed into stable curves
     rather than left as noisy raw histograms, since dividing sample
     data by a noisy correction would inject that noise into every
     peak.
  5. A purely geometric per-pixel solid-angle map is computed directly
     from the calibrated panel geometry (independent of counting
     statistics), used to correct for how much of the scattered beam
     each pixel actually catches.

Reading the output
-------------------
Everything below lives under
``/SNS/<Instrument>/shared/Vanadium/<Output Folder>/``. Two files are
the actual normalization inputs consumed by the **Normalization** tab
of the main reduction plan - everything else in that folder is a
diagnostic for checking the run, not something later steps read.

- ``flux.nxs`` - the background-subtracted, absorption-corrected
  incident spectrum, per bank, expressed as a cumulative distribution
  over wavelength. For a peak observed at a given wavelength on a
  given bank, this is what supplies "how much flux was actually
  available at that wavelength."
- ``solid_angle.nxs`` - the per-pixel vanadium response integrated
  over the full wavelength band: a single number per pixel folding
  together its detector efficiency and the geometric solid angle it
  subtends. This is what supplies "how strongly would this specific
  pixel respond."

Together, evaluating the flux at a peak's wavelength/bank and scaling
by the solid angle of the pixels it landed on is the full flat-field
correction described above.

Diagnostics only - the remaining files are written for visual
inspection of the run and are never read back in by the software:

- ``incident.nxs`` / the *incident* plot - the delivered spectrum
  shape versus wavelength, prior to the cumulative-flux conversion in
  ``flux.nxs``.
- ``count_rate.nxs`` / the *count_rate* plot - the per-bank detector
  efficiency curve; a tube or bank whose curve has a noticeably
  different shape from its neighbors usually indicates a mask or
  calibration issue worth revisiting.
- ``solid_angle_geom.nxs`` - the purely geometric solid-angle map
  (efficiency factored out); instrument coverage maps (gamma/nu
  scatter plots) are also saved for visually spot-checking masked
  regions or panels that look discolored relative to their neighbors.
- ``background.nxs``, ``background_count_rate.nxs`` - the standalone
  background run and its rate, useful for confirming the background
  run itself was clean.
- ``spectra.nxs``, ``correction.nxs``, ``scale.nxs`` - intermediate
  absorption-correction workspaces, useful for tracing a suspicious
  result back to a specific correction step.
- ``absorption_parameters.txt`` - the sanity-check report described
  in Step 1.
