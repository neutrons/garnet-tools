Generating Vanadium Normalization Files
=========================================

This tutorial walks through the **Vanadium** sub-tab (inside the
**Calibration** tab), which turns a vanadium run into the correction
files used to normalize Bragg peak intensities in the main reduction
pipeline.

The physical idea
------------------
A raw Bragg peak's integrated counts depend on more than the sample's
crystal structure: how many neutrons of that wavelength the source and
guide actually delivered that day, how efficiently the particular
pixels the peak landed on convert a neutron into a recorded count, and
how much solid angle those pixels subtend as seen from the sample.
None of that is physics you want in your final structure - it's
instrumental, and it has to be measured and divided out.

Vanadium is used to measure it because vanadium's coherent (Bragg)
scattering is almost zero. A vanadium sphere or rod in the beam
produces an almost purely elastic, incoherent signal with no crystal
structure of its own - a smooth, structureless response whose only
remaining wavelength- and pixel-dependence comes from exactly the
three instrumental effects above. That makes a vanadium run a flat
field: divide a real sample's data by it (after a few corrections) and
what's left over is the sample's own scattering.

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
  standard: only the illuminated volume contributes signal, and a
  beam that's smaller than the vanadium changes how much of the rod or
  sphere is actually seen.
- After running, check ``absorption_parameters.txt`` in the output
  folder: it reports the linear absorption/scattering coefficients and
  the equivalent mass and density it derived from your entered
  dimensions. Vanadium's density is well known (~6.11 g/cm\ :sup:`3`),
  so a wildly different implied density is a quick sign the entered
  diameter/height doesn't match the physical standard.

Step 2: Wavelength range and pixel grouping
----------------------------------------------
- **k(min)/k(max)** set the momentum (equivalently wavelength) window
  the correction will cover; this should bracket the full band you
  intend to use for the actual sample reduction; it auto-populates
  from the instrument's usable wavelength band when you change
  **Instrument**.
- **Rows/Cols** sum neighboring pixels together before building the
  correction curves. Vanadium runs are typically much shorter than
  sample runs, so grouping trades a little spatial resolution for a
  correction surface with far less counting-statistics noise - and
  it's applied consistently, so the same grouping should describe what
  you intend to do with your sample data downstream.
- **Mask Options** exclude dead or noisy tubes/pixels/banks. This
  matters even more here than for a sample run: after grouping,
  one bad pixel can poison the smoothed signal of its whole
  neighborhood, leaving a visible artifact in the final correction
  map.

Step 3: Run and inspect the correction
-----------------------------------------
- Point **Instrument Definition**, **Detector Calibration**, and
  **Tube Calibration** at the same geometry you calibrated in the
  previous tutorial - the vanadium correction has to be computed on
  the exact panel geometry that will be used to reduce the sample
  data, since it depends on each pixel's real position and solid
  angle.
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
     into two physically distinct pieces: the incident spectrum
     (how much flux the source delivered at each wavelength, summed
     over the whole array) and a per-bank detector efficiency curve
     (how sensitive each bank is as a function of wavelength, cross
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
- ``incident.nxs`` / the *incident* plot - the delivered spectrum
  shape versus wavelength, used later to normalize every peak by how
  much flux was actually available at its wavelength.
- ``count_rate.nxs`` / the *count_rate* plot - the per-bank detector
  efficiency curve; this is what removes "some banks read
  systematically higher than others" artifacts from the final data.
  Compare curves bank to bank - a tube or bank whose curve has a
  noticeably different shape from its neighbors usually means a mask
  or calibration issue worth revisiting.
- ``solid_angle_geom.nxs`` - the geometric solid-angle map; instrument
  coverage maps (gamma/nu scatter plots) are also saved so you can
  visually spot-check for masked regions or panels that look
  discolored relative to their neighbors.
- ``background_count_rate.nxs`` - the standalone background rate,
  useful for confirming the background run itself was clean.
- ``absorption_parameters.txt`` - the sanity-check report described
  in Step 1.

These files are then supplied as the vanadium normalization inputs to
the **Normalization** tab of the main reduction plan.
