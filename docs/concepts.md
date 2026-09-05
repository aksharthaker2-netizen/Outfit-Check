# Concepts & Design Log — Outfit Check

Running log of ML/CV concepts, design decisions, and — importantly — real
bugs found through testing, in the order they occurred. Kept honest on
purpose: limitations and mistakes are documented alongside what worked,
since that's the actual signal of engineering understanding.

---

## P1 — Pose Detection & Framing Check

**Concept:** MediaPipe Pose (heatmap-regression model) outputs 33 body
landmarks, each with a `visibility` confidence score. Framing validation
splits into two independently-checked failure modes: (1) low visibility
(occlusion/absence) and (2) landmark proximity to the frame edge (cropping)
— confidence alone doesn't catch a body part that's technically "seen"
right up until it's cut off by the frame boundary.

**Model choice:** MediaPipe Pose over YOLOv8-pose — per-landmark visibility
scores are the critical feature this task needs; YOLOv8-pose's confidence
is coarser and the model is heavier for a CPU-only live-guidance loop.

---

## P2 — Height Estimation & Body-Shape Classification

**Concept:** monocular height estimation requires reference-object
calibration to resolve scale ambiguity (a single 2D camera can't
distinguish "tall person far away" from "short person close up").
Calibration: `pixel_per_cm = reference_object_pixel_height / reference_object_real_height_cm`.

**Known limitation (v1):** requires the reference object and the person
to be at roughly the same distance from the camera. Confirmed in practice
— an early test placed the reference object on a bed (closer to camera)
while the person stood farther back, producing an implausibly short
height estimate. Root cause, not a bug: an inherent constraint of
single-camera measurement without depth sensing.

**Head-top extrapolation:** MediaPipe has no literal head-top landmark;
head-top y is extrapolated from the nose-to-eye vertical gap via a fixed
anthropometric ratio. Introduces a few cm of systematic error — flagged
as a known approximation, not treated as exact.

**Original body-shape classifier (3-category):** rectangle / pear /
inverted-triangle, from shoulder-width ÷ hip-width using MediaPipe
landmarks (11/12 shoulders, 23/24 hips) directly. Hourglass and apple
were not classifiable — no waist landmark exists in MediaPipe Pose.

---

## P3 — Garment Segmentation & Color Scoring

**Concept:** transfer learning via a pretrained segmentation model
(`mattmdjaga/segformer_b2_clothes`) rather than training from scratch —
standard practice when a pretrained model's label set already fits the
task. K-means clusters garment-mask pixels into dominant colors. Color
harmony scoring converts RGB → HSV and reasons about **hue** distance,
since harmony rules (analogous/complementary/triadic) are defined on the
color wheel, not raw RGB. Neutral colors (low saturation, near-black,
near-white) are treated as harmonizing with anything.

**Validated:** segmentation mask visually confirmed to correctly trace
garment boundaries (shirt, pants) against a cluttered/patterned
background, without bleeding onto skin or background.

---

## P4 — Person-Conditioned Scoring

**Concept:** the actual thesis of the project — the same garment colors
should score differently depending on the wearer's body shape. Modifier
architecture: `final_score = harmony_score × fit_factor`, where
`fit_factor` (bounded 0.85–1.15) encodes value-contrast placement rules
per body shape (pear: brighter top/darker bottom favored; inverted
triangle: reverse; rectangle: any strong contrast favored, direction
doesn't matter).

**Validated concretely:** same photo, same garment colors, scored under
all body-shape assumptions — produced a real 0.76–0.83 score spread
purely from changing the assumed body shape. This is the direct,
demonstrated proof of the person-conditioning thesis.

**Stated limitation:** only uses dominant-color brightness — says
nothing about garment fit, cut, or length, since no signal exists yet
for those.

---

## P5 — Rule-Based Recommendations

**Concept:** recommendations are grounded in the exact structured output
of the P4 scorer (harmony relationship + fit-factor reasoning), not
generated freely — avoids the failure mode of an LLM inventing
plausible-sounding advice disconnected from what was actually measured.
Scoped strictly to the two levers the scorer reasons about: color
harmony and value-contrast direction.

**Threshold-tuning finding:** an initial fit-factor penalty threshold
(0.97) was set as an arbitrary guess and silently suppressed a
legitimate recommendation for a real test case (fit factor landed
exactly on the boundary). Loosening to 0.99 (any value below the neutral
baseline of 1.0 surfaces a recommendation) fixed it. Concrete example of
threshold sensitivity in rule-based systems — caught only by running
against real data, not by code inspection.

---

## P2b — Waist-Width Estimation from Segmentation (major debugging round)

**Concept:** MediaPipe has no waist landmark, so waist width is estimated
from the garment segmentation mask instead — scanning horizontal rows of
the top-garment mask between shoulder-y and hip-y, taking the narrowest
row width as the waist proxy. This unlocks the full 5-category taxonomy
(adding hourglass and apple), since those require a narrow-waist signal
the shoulder/hip ratio alone can't provide.

### Bug 1 — landmark/mask methodology mismatch
MediaPipe's hip landmarks (23/24) sit at the **hip joint** (internal),
while shoulder landmarks (11/12) approximate the **outer shoulder edge**.
Using both directly produces a systematically inflated shoulder/hip
ratio. **Confirmed empirically:** landmark-only hip width measured 0.201
vs. segmentation-mask hip width of 0.367 — nearly double — on the same
photo. Fix: measure both shoulder and hip width from segmentation masks
(mask width at the landmark's y-row), not from landmark x-coordinates
directly. This flipped a real classification from `inverted_triangle`
(1.61) to `rectangle` (1.15) on the same photo.

### Bug 2 — partial-fallback mixing
When only one of the two mask-based measurements was available (e.g.
bottom-garment mask had no pixels at the hip-landmark row), the code
fell back to landmark-based hip width while still using mask-based
shoulder width — mixing methodologies again, producing a *more*
distorted ratio (2.27) than either pure method alone. Fix: all-or-nothing
fallback — use both mask measurements or neither.

### Bug 3 — boundary-artifact narrow-row picking
The "narrowest row in the shoulder-to-hip range" method is not robust to
non-waist narrow points at the scan boundaries — a boxy oversized shirt's
hem line (right at hip_y) registered as narrower than any genuine
mid-torso row, producing an anatomically implausible waist reading
(0.209 — under a third of shoulder width). **Confirmed via debug
instrumentation** (`row_used_y_normalized` landed at 0.703, essentially
equal to the hip_y boundary of 0.705). Fix: exclude a 15% margin from
both ends of the scan range before searching for the minimum, so
collar-area and hem-area artifacts can't be selected.

**Outcome of fixing all three bugs — the actual fitted-vs-baggy test:**
same person, fitted top vs. boxy/oversized top, both classified as
`rectangle` after fixes. Waist/hip ratio nearly identical (0.869 vs
0.867); waist/shoulder ratio moved in the expected direction (baggy
read as less waist-defined: 0.756 vs 0.814) but not enough to flip the
category in this case. **Honest finding:** the hypothesized
garment-looseness limitation is real and directionally confirmed, but
was smaller in practice than the three implementation bugs that had to
be fixed first to even measure it cleanly — testing surfaced more value
by finding those bugs than by confirming the original hypothesis.

**Known remaining gap:** single-row sampling (for waist, hip, and
shoulder) is inherently fragile to occlusion or unusual framing — one
test photo had zero mask pixels at the exact hip-landmark row (long top
occluding the waistline at that specific y), returning no hip-width
signal at all. A v2 improvement would sample a small window of rows and
take a median/mode rather than a single row.

---

## Known untested paths (as of this log)

- **Color harmony "clashing" classification:** logically verified via
  hue-band math (a hue distance falling outside all defined
  analogous/triadic/complementary bands), but never triggered against a
  real photo — no test case with the right hue separation (~35–100° or
  140–160°) has been tried yet.

## Follow-up: row-window fix (partial success + new finding)

Fix: `estimate_hip_width` / `estimate_shoulder_width` changed from single-row
lookup to a windowed median (±3% of frame height) — addresses occlusion at
exactly one row.

**Result on photo4 (the original motivating case): unresolved.** The bottom
mask still returns None even with windowing — confirmed via the debug
overlay that photo4's shirt hem extends so far past the hip landmark that
no pants are exposed anywhere within a ±3% window of hip_y. This is not a
windowing-size problem; a fixed landmark-derived y-row cannot work when the
top garment's actual extent varies this much between photos. Correctly
identified as a genuine limitation, not chased further with an arbitrarily
large window (which would start measuring semantically wrong regions).
Fallback to landmark-only classification remains the correct, safe behavior
in this case.

**Real v2 fix (not yet implemented):** detect the bottom mask's own topmost
row with any pixels, and use that as the measurement point instead of a
fixed hip_y — adapts to garment extent instead of assuming landmark
placement always lands on exposed fabric.

**Side effect on photo2:** windowing changed hip width slightly (0.367 →
0.380), which was enough to cross the WAIST_NARROW_RATIO_THRESHOLD (0.85)
boundary and flip classification from rectangle to hourglass. Confirms
threshold values (0.85/0.95) are sensitive to small measurement changes —
these are hand-picked, not tuned against real data, and should be
revisited once a proper multi-photo evaluation set exists.