# Dog Robot

Project for flock and sheepdog detection in aerial video, guidance reference generation, and robot simulation with MPC.

The project includes three main blocks:

1. Detection and tracking with YOLO.
2. Offline prediction of the target using only flock information.
3. Robot simulation with MPC on video.

## Objective

The goal is to move a simulated robot so that it behaves like a shepherd dog. To do that, the system can:

1. Generate a geometric guidance reference behind the flock.
2. Infer where the target should be by looking only at the flock motion.
3. Move the simulated robot toward that reference using MPC.

## Structure

```text
src/
  calculate_robot_guidance.py
  config_yolo.py
  extract_frames.py
  prepare_dataset.py
  simulate_robot_control.py
  simulate_robot_kinematics.py
  simulate_robot_mpc.py
  simulate_robot_mpc_dog.py
  track_flock_motion.py
  train_model.py
  flock_only_target_prediction/
    common.py
    model.py
    simulate_robot_mpc_flock_target_predictor.py
    train_flock_target_predictor.py
```

## Requirements

The project uses Python with these main dependencies:

```text
opencv-python
numpy
scipy
ultralytics
torch
```

Recommended installation:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install opencv-python numpy scipy ultralytics torch
```

## Detection Model

The main detector used in the project is:

```text
models/dogRobot_v2_best.pt
```

The class mapping expected by the code is:

1. `0: flock`
2. `1: dog`

## Classic Project Flow

### 1. Prepare the YOLO dataset

Split images and labels into `train`, `val`, and `test`.

```bash
python src/prepare_dataset.py --images "frames/images" --labels "frames/labels" --output "datasets/dog_robot" --train-ratio 0.8 --test-ratio 0.1
```

### 2. Train the YOLO detector

```bash
python src/train_model.py --data "datasets/dog_robot/dataset.yaml" --model yolo26s.pt --epochs 100 --image-size 960 --batch-size 8 --name "dogRobot_v2"
```

### 3. Track the flock

```bash
python src/track_flock_motion.py --video "videos/rebano_01_1min.mp4" --model "models/dogRobot_v2_best.pt" --confidence 0.15 --image-size 960 --name "flock_motion"
```

### 4. Generate the classic guidance reference

```bash
python src/calculate_robot_guidance.py --video "videos/rebano_01_1min.mp4" --trajectory "runs/tracking/flock_motion/rebano_01_1min_trajectory.csv" --target-x 1500 --target-y 300 --safety-margin 100 --goal-radius 80 --name "guidance_v1"
```

### 5. Simulate the robot with MPC

```bash
python src/simulate_robot_mpc.py --video "videos/rebano_01_1min.mp4" --guidance "runs/guidance/guidance_v1/rebano_01_1min_guidance.csv" --robot-start-x 1820 --robot-start-y 80 --initial-heading 0 --name "mpc_v5"
```

## Offline Pipeline: Flock-Only Target Prediction

This pipeline learns the relationship between flock motion and the expected dog/target position from multiple videos. It can then infer that target even when the dog is not visible.

Important:

- It uses a temporal CSV dataset with flock features and target positions as supervision.
- It learns only from flock features.
- The robot follows the predicted position using the same MPC base.

### 1. Train the flock-only predictor

```bash
python src/flock_only_target_prediction/train_flock_target_predictor.py --dataset "runs/flock_target_dataset/flock_target_dataset.csv" --history-length 12 --prediction-horizon-seconds 0.6 --epochs 60 --hidden-size 64 --num-layers 1 --device cpu --name "flock_target_gru"
```

Main output:

```text
models/flock_target_gru_flock_target_predictor.pt
```

### 2. Test it on a new video

```bash
python src/flock_only_target_prediction/simulate_robot_mpc_flock_target_predictor.py --video "videos/rebano_01_1min.mp4" --detector-model "models/dogRobot_v2_best.pt" --predictor-checkpoint "models/flock_target_gru_flock_target_predictor.pt" --robot-start-x 1820 --robot-start-y 80 --initial-heading 0 --predictor-device cpu --name "flock_target_test"
```

## How the Parts Connect

The control core is still `src/simulate_robot_mpc.py`.

That file provides:

1. `build_reference_horizon(...)`
2. `solve_mpc(...)`
3. `update_state(...)`

The new pipelines do not replace the MPC. They generate the `target` or `desired_x`, `desired_y` automatically, and the MPC then consumes those values.

## Generated Files

The repository ignores these by default:

1. `runs/`
2. `models/`
3. `videos/`
4. `datasets/`

So in practice, only source code is usually pushed to GitHub.

## Practical Notes

1. With few videos, the models are still prototypes and may generalize only in a limited way.
2. The `flock_only_target_prediction` predictor does not need to see the dog at inference time, but it depends strongly on training quality and variety.
3. The flock-only scripts can take a while because they first build guidance frame by frame with YOLO and then run the full simulation.

## Outputs

Each pipeline usually generates:

1. An `.mp4` video with English overlays.
2. A `.csv` file with robot, target, and MPC state.

Outputs are stored inside `runs/`.
