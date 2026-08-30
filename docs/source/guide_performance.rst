Speed and Memory: Grouping, Binning, and Symmetry
=====================================================

Covers three settings on the **Normalization** tab that most directly
control how long a reduction takes and how much memory it needs,
independently of the physical quality trade-offs already covered
elsewhere: pixel **Grouping**, the per-dimension **Bins**, and
**Symmetry**. None of these change what garnet is capable of
computing - only how expensive it is to compute it - so the guidance
here is about not paying for resolution or symmetry coverage you
don't actually need for the step you're on.

Pixel grouping
----------------
**What it does.** Before a run is ever converted to Q, garnet groups
neighboring detector pixels together (``Grouping``, e.g. ``2x2`` or
``4x4``) with ``GroupDetectors`` and immediately compresses the
result with ``CompressEvents``. This happens once, up front, on the
raw event workspace for that run - every later step (calibration,
masking, ``ConvertToMD``) then works on the grouped, coarser
workspace instead of the full per-pixel one.

**Why it's faster and lighter.** Grouping N x N pixels together cuts
the number of distinct spectra (and therefore the number of
individual event lists Mantid has to calibrate, mask, and convert)
by roughly N :sup:`2`, and ``CompressEvents`` collapses now-redundant
overlapping events on top of that. Since this happens before
``ConvertToMD``, every downstream operation - including the MDNorm
pass discussed below - is working with a smaller workspace for the
rest of that run's processing. For a reduction that loops over many
runs (a full experiment, or a survey/scan), this per-run load-and-
convert cost is usually where the bulk of the wall time actually
goes, so grouping is the single biggest lever for that part of the
pipeline.

**Where it applies.** ``Grouping`` is a **Normalization**-tab
setting only - it does not touch the separate **Integration** step,
which always works from full-resolution per-pixel data for peak
shape and position. That means grouping can be turned up aggressively
for normalization/slice-preview work without any effect on peak
integration precision elsewhere in the same plan.

**The trade-off.** Coarser grouping means coarser detector-space
resolution: peak positions and shapes in the raw data are less
precisely located, and the vanadium solid-angle/flux correction is
only as fine-grained as whatever grouping it was itself computed at
(see the :doc:`vanadium guide <guide_vanadium>`) - the grouping used
here must match the grouping the vanadium files were generated with,
or the correction and the data will disagree pixel-for-pixel. In
practice: use the coarsest grouping that still matches your vanadium
files while you're iterating on projections/extents/symmetry, and
only go finer (or to no grouping) for the final production pass if
detector-space resolution genuinely matters for that output.

Output binning
----------------
**What it does.** The three ``Bins`` values (one per projection axis,
``Uproj``/``Vproj``/``Wproj``) set the shape of the output data/norm
histogram that MDNorm actually builds and writes. Unlike grouping,
this doesn't shrink an intermediate workspace - it *is* the size of
the final result, and both the memory MDNorm needs while accumulating
it and the time it takes to fill every voxel scale with the total
voxel count, ``Bins[0] x Bins[1] x Bins[2]``. Garnet enforces a hard
cap of 1001\ :sup:`3` total bins specifically to keep that number
from running away.

**The three axes don't have to match.** It's tempting to set all
three bin counts equal (e.g. the GUI default of 201x201x201), but the
three axes are rarely equally informative. If you're looking at (or
integrating over) a thin or genuinely lower-resolution direction -
the vertical/out-of-plane axis of a 2D slice, or any direction whose
features you don't need to resolve finely - every extra bin along
*that* axis multiplies the total work by the same factor as an extra
bin in a direction that actually matters, for no analytical benefit.
As a rule of thumb: keep the low-value axis around 100 bins or fewer,
and reserve the higher resolution - 200 to 800 bins - for the one or
two in-plane directions where the reciprocal-space features you care
about (Bragg peaks, diffuse scattering) actually need resolving. Going
from, say, a uniform 400x400x400 grid to 400x400x100 for the same
in-plane detail cuts total voxels - and MDNorm's memory and run time
with them - by a factor of four, just by not over-resolving the axis
that didn't need it.

Symmetry operations
----------------------
**What it does.** The ``Symmetry`` parameter names a Laue point group
and is passed straight through to MDNorm's ``SymmetryOperations``.
Internally, MDNorm doesn't just tag the output with a symmetry label -
it rotates the normalization (and background, if present) by *every*
operator in that point group and re-bins/accumulates each rotated
copy into the same output histogram, so that the final normalization
correctly reflects what detector coverage looks like from every
symmetry-equivalent orientation, not just the one actually measured.

**The cost.** That means the normalization pass is repeated once per
operator: a low-symmetry Laue class (triclinic, 1-2 operators) costs
about the same as no symmetry at all, while a high-symmetry cubic
class like m-3m has 48 operators - so requesting full m-3m symmetry
can make the same bin grid take up to roughly 48 times longer to
normalize than leaving ``Symmetry`` unset. This multiplies directly
with the binning cost above: 48 operators over a needlessly fine
400x400x400 grid is far more expensive than 48 operators over the
same grid with its low-value axis trimmed to 100 bins.

**Guidance.** Don't pay the full symmetry cost while you're still
iterating - leave ``Symmetry`` unset (or use whatever minimal point
group you're confident in) while choosing projections, extents, and
bins, and only turn on the true, full Laue symmetry for the final
production normalization once the geometry is settled. Re-running a
48-operator normalization repeatedly during exploration is by far the
most expensive way to iterate.

Putting it together
----------------------
These three settings compound multiplicatively, so the order to tune
them in matters: pick a **Grouping** that matches your vanadium files
and is coarse enough for fast iteration, choose **Bins** that spend
resolution only on the axes that need it (~100 or fewer on the
low-value axis, 200-800 on the axes that matter), and leave
**Symmetry** off until everything else is settled - then turn full
symmetry on for the one production run that needs it.
