# Phase 13 — High-Magnification Revisit: Capabilities and Limitations

## What the system guarantees

1. **Frozen reference (§1)** — `FINALIZE SURVEY MAP` writes a self-contained
   bundle to `<session>/survey_final/`:
   `mosaic.png` (RGB stitch), `coverage_mask.png` (pixel coverage),
   `coverage_grid.png`, `monolayer_mask.png` + `monolayer_conf.npy`
   (semantic evidence), `placements.json` (every field pose + registration
   transform + source keyframe), `survey.json` (calibration, grid geometry,
   session metadata, this document's summary). The bundle is immutable;
   exploration never writes into it, and high-mag fields never enter the
   live survey map or mosaic.

2. **Display vs optical separation (§2)** — the map panel's zoom/pan is pure
   rendering (Phase 5 rule); the objective selector changes only the optical
   calibration used for physical units. No GUI zoom ever changes a
   coordinate.

3. **Calibration (§3)** — `configs/live.yaml → survey.objectives` stores
   `pixels_per_micron` per objective with an honest `source` flag
   (`estimated` until a stage-micrometer calibration is performed).
   Physical position = scan px × pixels-per-micron(reference objective).

4. **Coordinate transform (§4)** — a high-mag field is chain-registered
   against previous high-mag frames (translation, same mechanics as the
   survey chain), then expressed in survey coordinates:
   `pose_survey = anchor + chain_delta × (ppm_reference / ppm_high)`.
   The monolayer boundary therefore remains in the SAME physical slide
   coordinate system regardless of magnification.

5. **Monolayer-only navigation (§5)** — every explored field is classified
   `INSIDE MAPPED MONOLAYER / NEAR MONOLAYER EDGE / OUTSIDE MAPPED
   MONOLAYER` against the frozen mask, with a unit guidance vector and
   distance (µm) toward the nearest mapped monolayer cell when outside.

6. **Revisit points (§6)** — generated strictly inside the frozen monolayer
   mask: `grid` (evenly spread), `top_confidence` (highest-evidence cells),
   `centroid` (densest point). Points are listed in µm and drawn on the map;
   GO TO NEXT / click selects the target.

7. **Relocalization (§7)** — `RELOCATE HERE` rescales the live frame to the
   reference pixel size and template-matches it against the frozen mosaic.
   On success it re-anchors the exploration chain; on failure it reports
   `POSITION UNKNOWN` and assumes nothing.

## Hard limitations (read before trusting high-mag positions)

- **Image-only world coordinates.** The whole map lives in image
  registration space. Nothing ties it to the physical stage. Until motorized
  stage encoders are available, every pose is "where the images say we are",
  not "where the stage is".

- **Slide removal invalidates the map.** Taking the slide out and putting it
  back — even approximately — shifts/rotates the physical slide relative to
  every stored coordinate. All revisit poses become wrong by that offset.
  Rotation cannot be recovered by the current relocalizer (translation-only).

- **Relocalization is best-effort.** It needs the frozen mosaic to contain
  visually distinctive structure where you are looking; featureless or
  periodic regions (dense RBC fields look alike) can match the wrong
  location. The response gate (`relocate_min_response`) trades misses for
  false positives — a miss is always reported, a false match is not
  detectable in software alone. Cross-check with a revisit-point overlay.

- **Magnification must actually change the field of view.** The transform
  uses the calibration ratio; if the configured `pixels_per_micron` values
  are wrong (they ship as estimates), reported micron distances are wrong by
  the same factor, though map-internal geometry stays self-consistent.

- **Focus/illumination changes break template matching** (contrast, background
  color). Relocalize right after switching objectives with focus roughly set.

- **Chained poses drift.** Between re-anchors, poses accumulate the same
  registration drift as any chain (Phase 7 machinery). Re-anchor via
  RELOCATE HERE whenever uncertainty feels high.

**Long-term solution** (architectural, not fixable in software): motorized
stage with encoders. Stage coordinates would replace image-registration
coordinates as the pose source without rewriting the map or classification
system — `pose_survey = T_stage_to_survey(stage_x, stage_y)`, calibrated
once per objective by imaging a reference point.
