# memory.md — AI-Guided Microscope Slide Navigation: complete work log

> One-file record of everything done in this project, why each decision was made,
> what it measured, and what was learned the hard way. Last updated: 2026-10-01
> (commit `f901817`). Read top-to-bottom for the full story, or jump to a section.

---

## 1. Project objective

Automatically identify the **monolayer region of a peripheral blood smear** while a
microscope is being moved, and eventually guide the microscope/stage toward it.
Per-field decision: `TOO_THICK` / `MONOLAYER` / `TOO_THIN` / `UNCERTAIN`.

Progression of phases (each kept intact, never overwritten):

| Phase | Input | Status |
|---|---|---|
| 0. Environment & repo setup | — | done |
| 1. Static-image baseline (`cpsam_v2`) | 10 JPGs | done, frozen in `outputs/cpsam_v2_baseline/` |
| 2. Video prototype (V1–V9) | 1 MP4 sweep | done, `outputs/video_prototype_01/` |
| 3. Live camera perception GUI | USB microscope camera | done, validated; daily-use tool |
| 4. Calibration with human labels | M/T/N/U keys | **in progress — collect labels** |
| 5. Navigation control | motorized stage | not started (deliberately) |

---

## 2. Environment (why every pin exists)

- **Machine:** Windows 11 24H2, NVIDIA GTX 1080 Ti (11 GB, Pascal sm_61), USB microscope
  camera "MC910F USB2.0" (native 800×448 @ ~47 fps; 1280×720 works; 1920×1080 does NOT).
- **Conda:** Miniforge at `C:\Users\assad\Miniforge3n`; env **`cellpose_env`**,
  **Python 3.12.7** (deliberately NOT latest 3.12.14 — see SAC below).
- **Key packages (pinned for a reason):** torch 2.5.1+cu121 + torchvision 0.20.1+cu121
  (last cu121 line; supports the 1080 Ti; plain `pip install torch` on Windows is CPU-only!),
  cellpose 4.2.1.1 installed **editable** from the `external/cellpose` submodule,
  numpy 2.2.6, scipy 1.15.3, opencv-python-headless 4.11.0.86, pandas 2.2.3,
  scikit-image 0.25.2, PyYAML 6.0.2, imagecodecs 2025.3.30, tifffile 2025.3.30.
- **Smart App Control (SAC) is ON on this PC** (enforcing). It blocks binaries without
  Microsoft cloud reputation — brand-new package releases AND rare Python builds get
  `DLL load failed ... Application Control policy` errors. Lessons:
  - prefer widely-deployed (older) package versions over latest;
  - Python stdlib itself can be blocked: conda-forge 3.12.14's `_bz2.pyd` worked once,
    then got blocked hours later → swapped env to 3.12.7 (established build) → fixed;
  - verdicts can lapse; a retry after ~1 min sometimes suffices;
  - never disable SAC silently — it's an irreversible user decision.
- **PowerShell activation fix (2026-09-30):** `conda init` was done, but the execution
  policy was `Undefined` (= Restricted) so the conda profile hook never loaded and
  `conda activate` silently did nothing. Fix: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

---

## 3. Static-image baseline (Phase 1) — why and what

**Why:** prove Cellpose can segment RBCs in our microscope images before any
real-time ambitions. Uses `cpsam_v2` (Cellpose-SAM v2 checkpoint) with default
settings; results frozen in `outputs/cpsam_v2_baseline/` — never overwritten.

**Measured (10 images, 1920×1080):** RBC candidates 933–1118 (mean ≈ 1034),
coverage 0.34–0.39, normalized NN spacing 0.96–1.07, contact ratio 0.70–0.86,
merge suspects ≈ 0.25% of cells. This produced the **prototype count gate
950–1050** (explicitly NOT a validated threshold — FOV- and camera-specific).
Provisional update 2026-10-01: the operator’s real-camera fields judged monolayer
sit at ~940–1000 cells, so the live/video gate was moved to **920–1025** pending
calibration with M/T/N/U labels.

**Design rules established here (still in force):**
- Cellpose lives in `external/cellpose` as a git submodule — import only, never modify.
- Flags, not deletions: suspicious objects are flagged (`possible_small_artifact`,
  `possible_wbc`, `possible_merged_rbc`), never removed.
- All thresholds relative to a robust per-image median cell size — no absolute pixels.
- Watershed splitting of merged cells exists but is **disabled** (only flagged).

---

## 4. Video prototype (Phase 2, commits `13fdb70`, `08fe949`, `22f1a9b`)

**Why:** the final system receives a continuous stream, not 10 hand-picked JPGs.
Running full Cellpose on every frame of a 30 fps video is wasteful (consecutive
frames are nearly identical), so this phase built the **motion-aware,
coarse-to-fine architecture**: decode → quality check → motion estimation →
keyframe selection → fast screen → Cellpose → features → score → smoothing →
annotated video + CSVs.

**Input:** `Dataset/video_dataset.mp4` — 1920×1080, 35.77 fps, 7,036 frames, 196.7 s,
manual one-direction sweep (mostly fast movement, stationary periods up to 11.5 s,
brightness drift 120→151, defocused start, clumped networks mid-video, focused
separated fields only in the last third).

**What was built (`src/video/`, `scripts/process_video.py`, `configs/video.yaml`):**
- Phase-correlation motion estimation (480 px grayscale, sign-verified) — px-accurate dx/dy.
- Keyframe selection: accumulated displacement gate + gross blur guard +
  rate cap + stationary heartbeat. NEVER fixed-interval sampling.
- Fast screening (cheap occupancy/edge features) — recorded on every field but
  does not gate Cellpose yet (measure first, decide later).
- Valid central ROI (centroid rule), per-field features: density (FOV-normalised),
  coverage, normalized NN spacing, edge gap, contact ratio, close-neighbour
  distribution, graph clusters, 4×4 uniformity CV.
- Prototype Monolayer Score 0–1: weighted mix of 6 suitability components
  (density, coverage, spacing, low crowding, low clustering, uniformity) —
  all weights/ranges configurable, none medically authoritative.
- Multi-signal classification: TOO_THICK/TOO_THIN need ≥2 independent signals;
  count alone can never decide (proven: 3 fields with only ~679 cells were
  correctly TOO_THICK via cluster dominance + degree + non-uniformity).
- Temporal smoothing over ACCEPTED fields only (EMA α=0.4 + majority vote, window 5).

**Results (`outputs/video_prototype_01/`, GPU run):** 7,036 frames → **78 keyframes**
(2,184 stationary-skips, 4,195 rate-limit, 579 blur); classes raw 60 TOO_THIN /
13 UNCERTAIN / 3 TOO_THICK / 2 MONOLAYER (smoothed 62/12/4); counts 6→1003,
peak score 0.82 at t≈173 s; longest smoothed-MONOLAYER run 4 fields / 8.1 s.
The scan genuinely crossed regimes: sparse → clumped → monolayer-band → edge.
`results.csv` holds one row per RAW frame (the master stats table the project tracks).

**Benchmarks (`outputs/benchmarks/`):**
- Cellpose (cpsam_v2, float32 — the 1080 Ti has no native bfloat16): **10.9 s/field**
  at 1920×1080; 5.1 s @1280; 2.6 s @960. Counts within ±71 (~9%) and mean score
  within 0.025 across resolutions → reduced resolution is decision-equivalent.
- Feature extraction: 5.9 s mean detailed vs **0.57 s live** (the DT merge-suspect
  analysis costs 5.3 s and changes **zero** of 78 classifications) → merge
  diagnostics run offline-only; live path skips them.
- Architecture conclusion (measured, not assumed): full Cellpose can never run at
  30 fps → **fast screening + asynchronous Cellpose on accepted fields** now;
  lightweight student model later (this pipeline generates its training data).

---

## 5. Live camera perception GUI (Phase 3, commits `1af854a`, `5fe8ac2`)

**Why:** the supervisor pivot — from file processing to a **live perception system**:
`camera → latest-frame buffer → motion/quality gate → fast screen → async Cellpose
worker → features → score → smoothing → live GUI + session logs`. The GUI is our own
application (`src/ui/`, `src/capture/`, `src/live/`); the official Cellpose GUI is not
modified and the submodule stays untouched.

**What was built:**
- `src/capture/sources.py` — one `FrameSource` interface: `camera:<idx|url>`,
  `video:<path>` (regression replay), `image:<path>`; native-first connection,
  backend `auto|msmf|dshow`, frame-verified open, auto-reconnect, precise diagnostics.
- `src/live/controller.py` — motion states (MOVING/SETTLING/STATIONARY), settle
  frames, NEW_FIELD on ~8 RBC-diameters of accumulated displacement (auto-scaled
  to the actual camera width), heartbeat, MANUAL trigger; **stale-result rule**.
- `src/ui/workers.py` — Qt-free threads: capture worker (never blocks) + ONE
  Cellpose worker (model loaded once, latest-frame SLOT — no backlog by construction),
  live vs detailed feature modes.
- `src/ui/main_window.py` — PyQt6 GUI: microscope viewport with the project's
  overlay style (thin outlines, centroid dots, merge highlights, valid ROI),
  LIVE ANALYSIS panel (Cellpose ROIs kept separate from RBC candidates),
  **✓ MONOLAYER DETECTED** banner + green field highlight (only on smoothed,
  fresh, multi-feature MONOLAYER — never count alone), scan-history strip
  (position → score; the seed of the future slide map), hotkeys:
  SPACE pause · A analyze now · D ids · O outlines · L live/detailed ·
  S save frame · R reset · G record · **M/T/N/U human labels** · Q quit.
- Session logging: `outputs/live_session_*/` — accepted_fields.csv,
  raw_frames_light.csv, labels.csv (calibration dataset), runtime_summary.json,
  session_metadata.json, run.log.

**Validated end-to-end:** 35.8 fps capture sustained through continuous Cellpose
(7,037 frames — the feed never blocks); latest-frame policy supersedes stale
requests; parked-field test produced 3/3 fresh MONOLAYER fields → banner condition.
On the real microscope: 1021 RBCs classified fresh in ~5 s at 800×448.

---

## 6. Bugs found & fixed (the expensive lessons — do not re-learn these)

1. **Clock mismatch:** video sources emitted file-clock seconds while staleness
   compared `perf_counter` (hours since boot) → everything stale forever.
   Rule: ONE session clock everywhere; file time only inside replay pacing.
2. **Staleness metric:** use NET displacement (signed Σdx, Σdy), never path length
   (registration jitter accumulates as positive path and voids parked results).
3. **cv2.Laplacian** rejects float32 input with CV_64F output → cast to uint8.
4. **Worker exceptions** must be caught per field/unit — a live app must never die
   or hang silently (crash-resilient capture + analysis loops).
5. **Shutdown drain** must wait for pending fields AND model loading before quit
   (model load varies 3→30 s).
6. **PyQt6** ignores crashes in slots by default (prints and continues) → without an
   excepthook that quits, a crashed window leaves a HEADLESS process holding the
   camera. Fixed with `sys.excepthook` → clean quit. Also `QColor(hex, alpha=)` is
   invalid — build QColor then `.setAlpha()`.
7. **Camera exclusivity (the big one):** a UWP app holds the camera EXCLUSIVELY even
   after its window closes — `WindowsCamera.exe` lingered in Task Manager and blocked
   OpenCV on every backend. Fix ritual: end the holder process (python.exe zombies
   from crashed runs count too!), then the camera frees instantly. The app now also
   retries the initial open ~15 s (startup race between old/new instances).
8. **Never force unsupported camera modes:** this microscope grants native 800×448
   and 1280×720 but FAILS at 1920×1080 (→ 0×0 dead device). Open native-first;
   probe with `scripts/list_cameras.py --probe-resolutions 0:dshow`.
9. **Pixel thresholds don't survive resolution changes:** motion threshold and
   density are auto-scaled/normalised to a reference FOV (below).
10. **FOV density bug (2026-10-01):** the score's density range (450–560/Mpx) was
    calibrated on the 1080p camera; the 800×448 camera samples the same optical
    field with 5.8× fewer pixels, reading ~2710/Mpx → good fields scored ~0.53
    UNCERTAIN. Density is now reported per reference-FOV Mpx
    (`reference_frame: [1920, 1080]`): the user's real field went 0.527 → **0.774
    MONOLAYER**. 1080p pipelines unchanged.
11. **mp4v re-encoding destroys pale cells** (981→658 detected) — use lossless PNG
    stills for test fixtures.
12. **Piping python through grep hides tracebacks and buffers stdout** — debug with
    `python -u` writing to a log file.

---

## 7. Current state & how to run everything

```powershell
conda activate cellpose_env
python scripts\list_cameras.py                       # camera probe (both backends)
python scripts\live_app.py --source camera:0 --backend dshow   # the live GUI
python scripts\live_app.py --source video:Dataset\video_dataset.mp4 --cellpose-width 960
python scripts\process_video.py                      # offline video pipeline (V1-V9)
python scripts\segment_dataset.py --dataset <folder> --experiment-name <name>
python scripts\verify_environment.py                 # sanity check
```

Rules of daily use: close Windows Camera / other camera apps first (exclusive
access); only one FYP instance at a time; the app self-heals startup races
(~15 s retry) and releases the camera even on crashes.

**Repository layout:** `src/` (analysis, segmentation, postprocessing, evaluation,
visualization, utils, video, live, ui, capture), `scripts/`, `configs/`,
`Dataset/` (10 static JPGs + source video, mp4 gitignored), `annotations/`,
`outputs/` (experiments; mp4 recordings gitignored).

---

## 7.5 Manual capture workflow & concurrent job manager (Phase G, 2026-10-01)

**Why:** auto-analysis wastes GPU time on fields the operator doesn't care
about during manual calibration work. Manual capture became a first-class
workflow without removing auto mode (both feed the same job system).

**What was built:** CAPTURE & ANALYZE button (instant frame copy + enqueue;
the camera never freezes), Auto Analyze ON/OFF toggle, SAVE SNAPSHOT (no
Cellpose), an ANALYSIS JOBS panel (queued/processing/completed with class,
score, human label), a job viewer (click a completed job -> its own captured
snapshot with overlays; results NEVER drawn over the changed live view),
thicker ORANGE merge-suspect contours + orange centroid markers, and per-job
human labels (M/T/N/U on the selected job; `human_label` stored separately
from `predicted_class`).

**Architecture:** explicit `AnalysisJob`s in a `JobManager` (MANUAL > FORCED >
AUTO priority; manual jobs never superseded, auto keeps latest-frame policy;
queue bounded at 20 with loud refusal), processed by a two-stage pipeline:
N GPU Cellpose workers (one resident model each) -> bounded mask queue ->
M CPU feature workers (features/merge/score overlap inference). MANUAL/FORCED
jobs always get detailed analysis incl. DT merge suspects; AUTO stays on the
fast live path. Manual results never touch the temporal smoother or the
live-field state. CUDA OOM is caught per job (FAILED + message, session safe).

**Worker benchmark (4 fields @960px, GTX 1080 Ti):** 1 GPU worker: first
result 11.4 s, total 20.4 s, 3.1 GB; 2 workers: total 19.1 s (best), 6.3 GB;
3 workers: 23.3 s; 4 workers: 29.7 s, 12.0 GB (worse - compute contention).
**Recommendation: 1 GPU worker + 2 CPU feature workers** (shipped default);
more GPU workers cost VRAM and latency for no real throughput gain.

**Gross-occupancy thick signal (2026-10-01, evening):** a clumped-network
field (job #8) defeated segmentation - cpsam returned only 20 masks over a
visually massive region - so its count/coverage read as bogus thin signals
and it classified TOO_THIN. Fix: the cheap pre-Cellpose screen's dark-
foreground occupancy is now a classifier input - occupancy >= 0.55
(`thick_occupancy_min`, configurable) classifies the field TOO_THICK
outright (tagged "(screen)" in the job viewer), because a mostly-foreground
frame is thick regardless of what the failed count says (spec section 23:
extremely dense regions need "move away", not a count). Validated on six
field archetypes including the failing one; empty/sparse fields still thin,
monolayer unchanged. Occupancy is also stored per job (results CSV) for
calibration.

**Phase 0 - result synchronization (2026-10-01):** the sidebar could show a
stale live result while the viewport showed a selected job (two different
result objects - pure GUI state binding, no counting error). The panel now
has explicit contexts: **VIEWING: LIVE** vs **VIEWING: JOB #N** with a bold
label; selecting a completed job binds EVERY sidebar metric (counts, features,
score, class, timings) to that job's single result object; returning to LIVE
restores the live analysis. The live banner is hidden while a job is viewed.

**Phase 1 - persistent scan map (2026-10-07):** every analyzed field is now
stored as a spatial FOOTPRINT (x, y, w, h rectangle - not a dot) in the
cumulative-displacement coordinate system, via the new `src/live/scan_map.py`
(`ScanMap`, thread-safe, `scan_map.json` persisted per session after every
field; swappable later for real stage coordinates without touching the GUI).
The old scan strip was replaced by a real **SCAN MAP panel**: class-coloured
footprint rectangles (red/blue/amber, green for monolayer), human-label
ticks, and a cyan crosshair at the current live scan position; it remembers
all fields until Reset Scan (R). Regression: 2 video fields analyzed and
persisted with correct footprints; the occupancy fix proved itself live
(clumped intro fields now correctly TOO_THICK instead of TOO_THIN).

**Phase 2 - monolayer enter/exit state machine (2026-10-07):**
`src/live/nav_state.py` - four states (OUTSIDE/ENTERING/IN_MONOLAYER/
LEAVING) driven ONLY by the smoothed multi-feature Monolayer Score of fresh
AUTO fields, with HYSTERESIS: enter 0.65 > exit 0.45 plus 2-consecutive-
field confirmation on both sides (configs/live.yaml `navigation:`). Mid-band
scores hold the current state - no flicker. The GUI banner now permanently
shows the machine state (green ✓ CURRENTLY IN MONOLAYER / amber ENTERING /
orange ⚠ LEAVING / red OUTSIDE) and Reset Scan resets it. Every transition
is logged with timestamp + scan position (nav_transitions.csv) for Phase 4
boundary reconstruction. Bugs the tests caught: RLock needed (update() ->
view() self-deadlock), leaving-count was reset instead of incremented
(IN->LEAVING never fired), missing src/live/__init__.py, view-only
CaptureWorker arg order. Untracked user sessions
(live_session_20261001_*, live_p2_regression2) left for the user's commit.

**Phase 3 - persistent monolayer highlighting (2026-10-07):** the scan map
now carries a dedicated monolayer EVIDENCE layer on top of the intact
per-field footprints. Each MONOLAYER field joins the layer; overlapping
later fields either re-affirm (agreements) or contradict (disagreements)
existing footprints, weighted double for human labels. Demotion requires
>= 2 contradictions AND a majority - a single contradicting frame can never
erase territory, and restoration requires exceeding evidence (a human label
does it immediately). The panel draws the layer as a y-scanline RECTANGLE
UNION (`union_rects`): overlapping green fields merge into one region with
a single sampled outer boundary instead of dozens of rectangles. The live
viewport's current-field green border remains separate. Config:
`scan_map:` (contradictions_to_demote 2, overlap_min_fraction 0.25,
human_label_weight 2).

**Phase 4 - continuous monolayer boundary (2026-10-07):**
`ScanMap.monolayer_boundary()` builds a spatial confidence grid over the
active monolayer layer (cell = footprint width x 0.125; each cell
accumulates the evidence confidence of covering fields), thresholds it to a
binary mask, applies MODEST cleanup only (small-gap closing 2 cells,
tiny-island removal < 4 cells), and extracts cv2 outer contours converted
back to scan coordinates. The contour is drawn as a 3px green line on the
scan map (same transform as footprints), cached per survey version so it
expands incrementally as scanning continues. Validated: 4-field chain ->
one connected region spanning observed extents; incremental extension
(2450 -> 3150 px); isolated field = separate contour (no unjustified
bridging); L-shape reconstructed from data (no hard-coded U); contradicting
overlap creates no territory; 0.5-cell discretization documented.

**Phase 5 - zoom/scale independence (2026-10-07):** the scan map panel now
has ONE explicit view transform (widget = centre + (fit - centre)*zoom +
pan): all geometry lives in scan/world coordinates and never in UI pixels.
Wheel zooms, left-drag pans, [+]/[-] buttons, and a **Reset View** button
(restore fit - does NOT delete scan history; Reset Scan remains the only
survey wipe). Fit is centre-anchored so zoom-about-centre is a pure scale.
Verified: 50%/100%/200% views and 300/600/900 px widget sizes keep field,
monolayer layer, boundary and crosshair aligned within 1%; pan translates
the whole view consistently. Composition lesson: with pan applied after
zoom, zooming simply scales the existing pan (p' = p*ratio) - the
cursor-style formula double-counted.

**Phase 6 - multi-sweep reconstruction (2026-10-07):** START NEW SWEEP
button begins a new trajectory on the SAME map; if a completed job is
selected it ANCHORS the new sweep to that field's location (offset = anchor
map pos - current raw position), so sweep-2 fields land correctly in the
sweep-1 frame without rewriting the coordinate system. `ScanMap` stores
sweep records (id, direction auto-detected from dominant motion, start
position, anchor job, n_fields) plus per-field sweep_id; footprints carry
map_x/map_y; the panel draws per-sweep dashed trajectories + numbered start
markers under the territory layers. Monolayer layer/boundary are NEVER
reset between sweeps - they accumulate across sweeps. All of it persists
in scan_map.json and dies only with Reset Scan. Stage-coordinate
replacement later = swap the offset/position source in ScanMap only.

**Launch-freeze bug (2026-10-08, found via user session log):** the GUI
froze on the first AUTO/manual job because AnalysisJob never received the
map_x/map_y fields - the Phase 6 patch that added them silently no-opped
(non-consecutive anchor text, no assert) while the job-submission code DID
pass them -> TypeError -> capture thread died at frame 10 -> frozen feed,
dead buttons, 3 fps of a static frame. Fixes: fields added (with assert);
a dead capture thread now shows 'CAPTURE STOPPED: <reason>' in the viewport
instead of a silent freeze. Audit of all Phase 0-6 tick paths (profiled at
11.5 ms/render tick offscreen) confirmed no GUI-thread starvation - the
3 fps was the frozen-pipeline symptom. Also fixed in the same sweep:
missing nav_machine param on MainWindow (first real-launch crash) and the
mono_rects rename leftover in ScanMapPanel.set_data (NameError on empty
survey startup). Lesson: EVERY scripted patch gets an assert; tests must
use the exact production call signatures.

**Phase 7 - robust frame registration & global pose (2026-10-08):**
`src/live/registration.py` - GlobalPoseTracker: translation-only phase
correlation on prepped (gray/downscaled/normalized, Hanning-windowed)
frames with ORB+median-translation VERIFICATION on every field (phase corr
WRAPS beyond half-frame and aliases large moves to confident wrong small
offsets - ORB is wrap-free and wins when its evidence is strong: matches
>= 15, inlier >= 0.5, conf >= 0.3). Confidence states GOOD/WEAK/FAILED;
overlap-consistency NCC gate catches wrapped garbage and featureless
frames (raw std < 3). Reference selection: preferred anchor (START NEW
SWEEP) > last GOOD > recent; loop-closure snap against nearby map fields
corrects drift; disagreements > 300 px -> POSE_DRIFT_WARNING. FAILED
fields are analyzed+logged but never enter the trusted map; GUI shows
'REGISTRATION FAILED - return to previous known field'. Reg metadata
(state/method/conf/dx/dy/ref/drift) flows to job objects, accepted_fields
CSV and the live HUD. Stage coordinates later = swap register_field's
position source only. Real-video regression: field 1 FIRST_FIELD GOOD;
field 2 (large move) correctly FAILED out of the trusted map.

**Phase 8 - coverage/overlap controller (2026-10-08):** `src/live/coverage.py`
- `evaluate_coverage(candidate_rect, existing, ...)` - exact y-scanline union
  overlap vs the scanned territory, best-single-field overlap, order-
  independent, works horizontal/vertical/diagonal; states ALREADY_SCANNED
  (>= duplicate_overlap_threshold 0.90), TOO_MUCH_OVERLAP (> target_max
  0.30), GAP_RISK (< target_min 0.20), CAPTURE_READY (in band),
  FIRST_FIELD (empty map). Config: configs/live.yaml `coverage:`.
- GlobalPoseTracker.preview_coverage(): live read-only overlap preview
  (phase corr vs last GOOD field, no commitment) for operator guidance.
- CAPTURE button restyles by guidance state (green READY / amber TOO_MUCH /
  dark red GAP / grey ALREADY); CAPTURE is never blocked - captures against
  guidance are recorded capture_type=FORCED in the CSV.
- AUTO duplicate skip: >= 90% overlap with confidently-registered territory
  skips Cellpose (skipped_duplicates counter); manual always runs.
- Coverage metrics per accepted field in accepted_fields.csv: capture_type,
  coverage_state, overlap_prev/total, new_coverage, duplicate_pct.
- GUI: live guidance line (arrow + scanned-overlap %) under the fast-screen
  line.

**Phase 9 - incremental RGB mosaic (2026-10-08):** `src/live/mosaic.py`
- MosaicBuilder: world-coordinate RGB canvas, dynamically expanding in any
  direction (floor/ceil integer alignment so placements never drift);
  per-pixel weight = rectangular edge feather x registration confidence;
  blend = weighted running average (incremental - never rebuilt from
  scratch); coverage mask = accumulated weight > threshold
- semantic colors stay on the scan map; the mosaic is pure RGB; placements
  recorded per field (job id, pose, size, conf, timestamp); exports:
  mosaic.png + coverage_mask.png + placements.json per session
  (mosaic_exports/); npz save/load for reproducibility
- GUI: stitched mosaic is the mini-map BASE LAYER (under trails/footprints/
  boundary), placed through the Phase 5 view transform (zoom-independent);
  mini-map also shows the current FOV rectangle; Reset Scan clears it
- FeatureWorker blends each registered field (conf-weighted, FAILED reg
  excluded); grayscale frames auto-converted
- tests: placement bounds, negative expansion, blending, exports,
  persistence, overview scaling, GUI layer render - all passed

**Phase 10 - gap & duplicate-revisit detection (2026-10-08):**
`ScanMap.coverage_grid()/coverage_summary()` - rasterized observation grid
(SCANNED=2 / LOW_CONFIDENCE=1 / UNSCANNED=0, cell = footprint width x
0.125, best-reg-conf per cell) over the survey bbox; coverage pct excludes
the 1-cell padded ring; INTERIOR UNSCANNED HOLES = unscanned components
surrounded by scanned territory (border-touching = outside, filtered by
min_hole_cells). Mini-map draws holes as red translucent + dashed-outline
'UNSCANNED GAP' markers. Duplicate/revisit: capture-time coverage preview
tags jobs with duplicate_of (best-overlap job) and revisit=True when
captured over scanned territory; both observations preserved in
accepted_fields.csv (capture_type/revisit/duplicate_of columns) - evidence
fusion stays conservative (Phase 3 layer already never overwrites).
Semantic boundaries still never cross unscanned space (monolayer union has
no interpolation; boundary closing is only 2 cells).

**Phase 12 - raster/multi-sweep navigation assistant (2026-10-08):**
`src/live/raster.py` - RasterNavigator: boustrophedon row tracking
(LEFT_TO_RIGHT/RIGHT_TO_LEFT, alternating per row; row parity chosen for
the row ABOUT to start), two phases (ROW_SWEEP: horizontal overlap in the
20-30% band; STEP_DOWN: vertical overlap vs the previous row in the
15-30% band), operator hints (MOVE RIGHT/LEFT/DOWN FARTHER/BACK, CAPTURE
READY), and note_vertical_step() flipping direction when a new row lands
in the vertical band. START NEW SWEEP drives it (scan_map sweep records
gain row/row_direction); capture worker preview carries raster hints into
the HUD (RASTER: ... line); configs/live.yaml `raster:` section. Motors
never controlled - human assistant only.

## 8. What's next (in order)

1. **Collect human labels** with M/T/N/U during real sessions — the calibration
   dataset (frames + full feature vectors + human class). No banner depends on it;
   it's purely for Phase 4.
2. **Calibrate** the prototype count gate (950–1050) and score weights from those
   labels — replacing every "experimental prototype" value in the configs.
3. Test `--cellpose-mode hybrid` (screen-gated Cellpose) once labels exist.
4. Multiple sweeps / slide-map panel (scan history already records position → score).
5. Only after live classification is stable: navigation control (out of scope today).
