# Dog Robot

Proyecto para deteccion de rebano y perro pastor en video aereo, generacion de referencias de guiado y simulacion de un robot con MPC.

El proyecto incluye tres bloques principales:

1. Deteccion y tracking con YOLO.
2. Prediccion offline del perro trasero o del target usando varios videos.
3. Simulacion del robot con MPC sobre el video.

## Objetivo

El objetivo es mover un robot simulado para que se comporte como un perro pastor. Para ello, el sistema puede:

1. Seguir directamente al perro trasero detectado.
2. Predecir su posicion futura con un modelo entrenado offline.
3. Inferir donde deberia estar ese target viendo solo el movimiento del rebano.

## Estructura

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
  rear_dog_prediction/
    build_rear_dog_tracking_dataset.py
    common.py
    model.py
    simulate_robot_mpc_trained_predictor.py
    train_rear_dog_predictor.py
  flock_only_target_prediction/
    common.py
    model.py
    simulate_robot_mpc_flock_target_predictor.py
    train_flock_target_predictor.py
```

## Requisitos

El proyecto usa Python con estas dependencias principales:

```text
opencv-python
numpy
scipy
ultralytics
torch
```

Instalacion recomendada:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install opencv-python numpy scipy ultralytics torch
```

## Modelo de deteccion

El detector principal usado en el proyecto es:

```text
models/dogRobot_v2_best.pt
```

Las clases esperadas por el codigo son:

1. `0: flock`
2. `1: dog`

## Flujo clasico del proyecto

### 1. Preparar dataset YOLO

Divide imagenes y etiquetas en `train`, `val` y `test`.

```bash
python src/prepare_dataset.py --images "frames/images" --labels "frames/labels" --output "datasets/dog_robot" --train-ratio 0.8 --test-ratio 0.1
```

### 2. Entrenar detector YOLO

```bash
python src/train_model.py --data "datasets/dog_robot/dataset.yaml" --model yolo26s.pt --epochs 100 --image-size 960 --batch-size 8 --name "dogRobot_v2"
```

### 3. Hacer tracking del rebano

```bash
python src/track_flock_motion.py --video "videos/rebano_01_1min.mp4" --model "models/dogRobot_v2_best.pt" --confidence 0.15 --image-size 960 --name "flock_motion"
```

### 4. Generar referencia de guiado clasica

```bash
python src/calculate_robot_guidance.py --video "videos/rebano_01_1min.mp4" --trajectory "runs/tracking/flock_motion/rebano_01_1min_trajectory.csv" --target-x 1500 --target-y 300 --safety-margin 100 --goal-radius 80 --name "guidance_v1"
```

### 5. Simular el robot con MPC

```bash
python src/simulate_robot_mpc.py --video "videos/rebano_01_1min.mp4" --guidance "runs/guidance/guidance_v1/rebano_01_1min_guidance.csv" --robot-start-x 1820 --robot-start-y 80 --initial-heading 0 --name "mpc_v5"
```

## Pipeline offline: rear dog prediction

Este pipeline aprende con varios videos la trayectoria del perro que va detras del rebano y usa esa prediccion como target del robot.

### 1. Construir dataset temporal

```bash
python src/rear_dog_prediction/build_rear_dog_tracking_dataset.py --videos "videos/rebano_01_1min.mp4" "videos/rebano_02.mp4" "videos/rebano_03.mp4" "videos/rebano_04.mp4" --model "models/dogRobot_v2_best.pt" --motion-window 12 --motion-dead-zone 10 --rear-projection-threshold 15 --lateral-penalty 0.35 --name "rear_dog_dataset"
```

Salida principal:

```text
runs/rear_dog_prediction/rear_dog_dataset/rear_dog_tracking_dataset.csv
```

### 2. Entrenar predictor offline

```bash
python src/rear_dog_prediction/train_rear_dog_predictor.py --dataset "runs/rear_dog_prediction/rear_dog_dataset/rear_dog_tracking_dataset.csv" --history-length 12 --prediction-horizon-seconds 0.6 --epochs 60 --hidden-size 64 --num-layers 1 --device cpu --name "rear_dog_gru"
```

Salida principal:

```text
models/rear_dog_gru_rear_dog_predictor.pt
```

### 3. Probarlo en un video nuevo

```bash
python src/rear_dog_prediction/simulate_robot_mpc_trained_predictor.py --video "videos/rebano_01_1min.mp4" --detector-model "models/dogRobot_v2_best.pt" --predictor-checkpoint "models/rear_dog_gru_rear_dog_predictor.pt" --robot-start-x 1820 --robot-start-y 80 --initial-heading 0 --motion-window 12 --motion-dead-zone 10 --rear-projection-threshold 15 --lateral-penalty 0.35 --predictor-device cpu --name "trained_predictor_test"
```

## Pipeline offline: flock-only target prediction

Este pipeline aprende con varios videos la relacion entre el movimiento del rebano y la posicion esperada del perro/target. Luego puede inferir ese target aunque el perro no sea visible.

Importante:

- usa como supervision el CSV temporal generado por `rear_dog_prediction`
- aprende solo con features del rebano
- el robot sigue la posicion predicha con el mismo MPC base

### 1. Entrenar predictor flock-only

```bash
python src/flock_only_target_prediction/train_flock_target_predictor.py --dataset "runs/rear_dog_prediction/rear_dog_dataset/rear_dog_tracking_dataset.csv" --history-length 12 --prediction-horizon-seconds 0.6 --epochs 60 --hidden-size 64 --num-layers 1 --device cpu --name "flock_target_gru"
```

Salida principal:

```text
models/flock_target_gru_flock_target_predictor.pt
```

### 2. Probarlo en un video nuevo

```bash
python src/flock_only_target_prediction/simulate_robot_mpc_flock_target_predictor.py --video "videos/rebano_01_1min.mp4" --detector-model "models/dogRobot_v2_best.pt" --predictor-checkpoint "models/flock_target_gru_flock_target_predictor.pt" --robot-start-x 1820 --robot-start-y 80 --initial-heading 0 --predictor-device cpu --name "flock_target_test"
```

## Como se conectan las partes

La base del control sigue siendo `src/simulate_robot_mpc.py`.

Ese archivo aporta:

1. `build_reference_horizon(...)`
2. `solve_mpc(...)`
3. `update_state(...)`

Los pipelines nuevos no sustituyen el MPC. Lo que hacen es generar automaticamente el `target` o `desired_x`, `desired_y` que luego consume el MPC.

## Archivos generados

El repositorio ignora por defecto:

1. `runs/`
2. `models/`
3. `videos/`
4. `datasets/`

Por tanto, normalmente se sube a GitHub solo el codigo fuente.

## Notas practicas

1. Con pocos videos, los modelos son prototipos y pueden generalizar de forma limitada.
2. El predictor `rear_dog_prediction` necesita deteccion del perro en inferencia.
3. El predictor `flock_only_target_prediction` no necesita ver al perro en inferencia, pero depende mucho de la calidad y variedad del entrenamiento.
4. Los scripts flock-only y offline pueden tardar bastante porque primero generan guidance frame a frame con YOLO y luego ejecutan la simulacion completa.

## Resultados

Cada pipeline genera normalmente:

1. un video `.mp4` con overlays en ingles
2. un `.csv` con el estado del robot, target y MPC

Las salidas se guardan dentro de `runs/`.
