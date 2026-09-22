# MPC Herding

Standalone implementation of the main algorithmic components from the provided PDF:

- Collective animal dynamics from Eq. (2)-(4).
- Self-organization rules: `ran`, `nn`, `ha`, `lch`, `nla`, `vicsek`, `boids`.
- Collective Motion Flow Field (CMFF) using Gaussian kernel smoothing, Eq. (6)-(8).
- Escape-region selection and maximum-divergence point, Eq. (9)-(10).
- Dynamic driving point with escape/herd direction fusion, Eq. (11)-(13).
- Nonlinear MPC with SLSQP, control bounds, rate constraints, Ackermann-derived angular-rate limit, and the cost terms from Eq. (16)-(18).

Run a smoke simulation from the repository root:

```powershell
python -m src.mpc_herding.simulation --herd-size 20 --max-steps 80 --horizon 5 --output-csv runs/mpc_herding/smoke.csv
```

This module is intentionally separate from the older video/YOLO scripts. It includes both a metric simulation mode and a video mode that uses YOLO flock detections.

## Video Flock-Only Mode

To process a real video with `models/dogRobot_v2_best.pt` and simulate the dog/robot using only the detected flock:

```powershell
python -m src.mpc_herding.video_flock_only --video path/to/video.mp4 --draw-detected-dog
```

The detector uses class `0` (`flock`) for control. Class `1` (`dog`) is only drawn when `--draw-detected-dog` is enabled and is not used by the controller.

The video overlay is in English. It draws the flock ellipse, flock center, target, driving point, simulated dog trajectory, and optionally detected real dogs. The flock bounding rectangle and the synthetic particles inside the ellipse are intentionally hidden.

### Guidance Modes

`--guidance-mode adaptive-sweep` is the default and recommended mode for real videos where the detector returns one global flock box/ellipse.

- `adaptive-sweep`: places the driving point behind the flock relative to the target and shifts it laterally according to the observed lateral drift of the flock center.
- `sweep`: places the driving point behind the flock and adds a periodic side-to-side sweep.
- `rear`: places the driving point behind the flock with no lateral sweep.
- `cmff`: uses the CMFF maximum-divergence point. This is closest to the PDF formulation, but it is unstable when the video only provides one global flock detection instead of individual animal positions.

For video use, `adaptive-sweep` is more stable than pure CMFF because the current detector provides a flock-level ellipse, not individual sheep positions and velocities.

### Recommended Command

```powershell
python -m src.mpc_herding.video_flock_only --video path/to/video.mp4 --model models/dogRobot_v2_best.pt --draw-detected-dog --guidance-mode adaptive-sweep --horizon 3 --synthetic-animals 80 --grid-resolution 21 --image-size 640 --driving-offset 70 --sweep-amplitude 100 --lateral-drift-gain 1.2 --robot-max-speed 180 --guidance-smoothing 0.9 --guidance-max-step 35
```

Use `--target x,y` if the default target location is not appropriate for the video:

```powershell
python -m src.mpc_herding.video_flock_only --video path/to/video.mp4 --model models/dogRobot_v2_best.pt --target 1850,350 --guidance-mode adaptive-sweep
```

You can also use the explicit form `--target-x 1850 --target-y 350`.

### Tuning The Pink Driving Point

- `--driving-offset`: distance from the ellipse border to the driving point. Lower values place it closer to the flock.
- `--sweep-amplitude`: maximum lateral displacement in `adaptive-sweep` and `sweep` modes.
- `--lateral-drift-gain`: responsiveness to the flock's lateral drift in `adaptive-sweep` mode.
- `--guidance-smoothing`: temporal smoothing of the driving point.
- `--guidance-max-step`: maximum per-frame displacement of the driving point.

Examples:

```powershell
# Closer to the ellipse
python -m src.mpc_herding.video_flock_only --video path/to/video.mp4 --driving-offset 40

# More lateral correction
python -m src.mpc_herding.video_flock_only --video path/to/video.mp4 --sweep-amplitude 150 --lateral-drift-gain 2.0

# More stable point
python -m src.mpc_herding.video_flock_only --video path/to/video.mp4 --guidance-smoothing 0.95 --guidance-max-step 20
```

### Faster Test Run

Use this to check that the pipeline works before processing the full video:

```powershell
python -m src.mpc_herding.video_flock_only --video path/to/video.mp4 --model models/dogRobot_v2_best.pt --max-frames 20 --horizon 1 --synthetic-animals 25 --grid-resolution 15 --image-size 416 --progress-interval 1 --debug-timing
```

### CMFF-Style Experimental Mode

```powershell
python -m src.mpc_herding.video_flock_only `
  --video path/to/video.mp4 `
  --model models/dogRobot_v2_best.pt `
  --guidance-mode cmff `
  --synthetic-animals 100 `
  --sigma 45 `
  --tau 0.0001 `
  --eta 0.5 `
  --horizon 8 `
  --output-dir runs/mpc_herding
```

Outputs:

- Annotated MP4 with flock ellipse, flock center, target, driving point, detected real dog boxes if requested, and simulated dog trajectory.
- CSV with per-frame robot state, target, flock center, driving point, optional CMFF fields, and optimizer status.
- By default, outputs are written to `runs/mpc_herding/`.

### Current Limitations

- The video mode does not make the real flock respond to the simulated dog; it overlays a simulated dog trajectory on prerecorded flock motion.
- `adaptive-sweep`, `sweep`, and `rear` are stable video heuristics, not the exact CMFF driving-point equation from the PDF.
- Exact CMFF requires individual animal positions and velocities. With only a flock-level detection, CMFF must be approximated and can be noisy.
- Coordinates are in image pixels, not calibrated meters, unless an external camera-to-ground calibration is added.
