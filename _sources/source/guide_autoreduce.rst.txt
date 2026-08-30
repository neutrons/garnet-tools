Live Autoreduction
====================

Covers ``garnet.utilities.autoreduce``, the script the facility's
post-processing pipeline runs automatically on every event NeXus file
as it lands, and the **Save to Autoreduce** button (on the main
reduction plan tab) that opts an experiment into its live slice
preview. Unlike the calibration and vanadium guides, nothing here is
run by hand during normal use - it happens unattended, per run,
without anyone opening garnet.

The physical idea
------------------
A full offline reduction (peak search, integration, normalization)
takes real setup and can only run once an experiment's UB and
normalization files exist. But there is a lot of value in a fast,
automatic sanity check on *every single run* the moment it finishes
collecting: is the detector seeing where it should, is this run even
still on the same crystal orientation as the last one, is the
beamline behaving. Autoreduction exists to answer that cheaply and
immediately, and - once the experiment has a UB and vanadium
normalization available - to keep a running, live reciprocal-space
preview updated run by run without anyone having to trigger it.

It is invoked as ``autoreduce.py <path-to-run>.nxs.h5``, with the raw
NeXus filename supplied by the facility's automated post-processing
service. Everything it produces is written next to that run, under
that experiment's ``IPTS-XXXX/shared/autoreduce/`` directory.

Stage 1: Compress to "lite" data
----------------------------------
Every run is first regrouped into a coarser, lighter workspace used
by the rest of autoreduction (and optionally by downstream tools):

- Neighboring pixels are summed together per the instrument's
  configured ``Grouping`` (e.g. 2x2), and the events are compressed -
  the same trade of spatial resolution for a smaller, faster
  workspace used for the vanadium correction (see the
  :doc:`vanadium guide <guide_vanadium>`).
- The regrouped workspace is given its own smaller instrument
  definition (looked up from ``/SNS/software/scd/lite/``) so it
  carries a geometry that actually matches its coarser pixel map.
- Standard edge/tube/bank masking is reapplied at the coarser
  grouping, using the same ``MaskEdges``/``MaskLost``/``MaskBanks``
  configuration as the rest of garnet for that instrument.
- The result is saved as a ``.lite.nxs.h5`` file alongside the run.

Stage 2: Instrument coverage plot
------------------------------------
The lite workspace is binned into a 2D gamma/nu (detector-space)
heatmap of mean counts per pixel - a quick "does the detector array
look right" picture: dead regions, a miscentered beam, or an
unusually hot panel are visible at a glance, well before any peak
finding or indexing happens.

Stage 3: Cross-correlation (CORELLI only)
--------------------------------------------
CORELLI's chopper requires an elastic cross-correlation to recover a
sharp time-of-flight signal from its otherwise pseudo-statistically
chopped beam. Autoreduction runs that correlation on every run and
saves the elastic-filtered result (also compressed to lite) - a
failure here is logged rather than raised, since not every run is
expected to have a clean elastic peak to correlate against.

Stage 4: Live slice preview (opt-in)
----------------------------------------
This stage only runs once an experiment has been opted in, and is
skipped silently otherwise.

**Opting in.** From the main reduction plan, once **IPTS** and a
**UB** file are set, click **Save to Autoreduce**. This copies the UB
file and writes a small trigger YAML (``UBFile``, ``VanadiumFile``,
``FluxFile`` - the latter two taken from whatever is already
configured on the **Normalization** tab) into that experiment's
``shared/autoreduce/`` directory. Autoreduction looks for the most
recently modified such YAML on every run; if none is found, or it is
missing any of those three keys, the live preview is skipped and only
Stages 1-3 run. Re-running **Save to Autoreduce** later (e.g. after
re-indexing to a better UB) simply writes a newer YAML, which is
picked up automatically on the next run.

**What it computes.** For each run, once a config is found:

1. The UB from the trigger YAML is attached and the goniometer is set
   from the run's own logged motor angles (not the generic
   "Universal" convention - TOPAZ and similar log motors under PV
   names that Mantid's default goniometer setup does not recognize).
2. Four default HKL projections are evaluated: the three principal
   zones (``hk0``, ``h0l``, ``0kl``), plus - only when the lattice
   isn't close to axis-aligned in the lab frame - one extra
   "equatorial" zone chosen the same way the reduction plan's
   **Autoproj** picks one, so a non-orthogonal cell still gets one
   informative wide-angle slice instead of three edge-on ones.
3. Each projection's in-plane extent is sized from a fixed
   :math:`d_\mathrm{min}` (configurable in the trigger YAML, default
   0.7 Å) and binned at a fixed 0.05 rlu step; the third, thinned axis
   is integrated over a fixed ±0.1 rlu window.
4. The run is normalized against the experiment's configured vanadium
   solid-angle and flux files via MDNorm - the same normalization
   used in the offline pipeline (see the
   :doc:`vanadium guide <guide_vanadium>`), just applied slice by
   slice as each run comes in rather than once at the end.

**Accumulation.** Runs sharing the same title (ignoring a trailing
run-enumeration suffix, e.g. ``NaCl_1``, ``NaCl_2`` -> ``NaCl_*``) are
treated as repeat measurements of the same setup and accumulated
together on disk, run over run, rather than each overwriting the
last. A change in UB (a different crystal mounted, or a fresh
orientation) changes the computed extents and therefore the
accumulation file's key, so a UB change starts a new file instead of
silently mixing old and new orientations together.

Publishing
-----------
Every plot generated across all four stages is collected into one
HTML page and, where the optional ``plot_publisher`` package is
available, uploaded to the facility's live monitoring display for
that instrument and run - the same page instrument scientists watch
during an experiment to catch problems as they happen rather than
after the fact.

Data confirmation
-------------------
A separate, related script, ``garnet.utilities.confirm``, is run by
the same automated pipeline after autoreduction to report each run's
processing status back to the facility's data-confirmation system
(``confirm-data``): it scans that IPTS's
``shared/autoreduce/reduction_log/`` directory for per-run ``.log``/
``.err`` files and reports ``Yes`` (all runs processed cleanly),
``Partially`` (some failed), ``No`` (all failed), or ``Unknown`` (no
logs found yet).

Reading the output
--------------------
Everything lives under ``IPTS-XXXX/shared/autoreduce/``, alongside
the raw run it was generated from:

- ``<run>.lite.nxs.h5`` - the regrouped, masked "lite" workspace
  (Stage 1).
- ``<run>_elastic.lite.nxs.h5`` - the cross-correlated elastic data,
  CORELLI only (Stage 3).
- ``<title_key>_<projection>_<geom_key>_data.nxs`` /
  ``..._norm.nxs`` - the accumulating MDNorm data/normalization pair
  for one HKL projection of one title group (Stage 4); dividing data
  by norm gives the normalized slice. These are what later runs of
  the same title load and add into.
- The published monitor page itself is not saved locally - it is
  uploaded directly to the facility's live monitoring display.
