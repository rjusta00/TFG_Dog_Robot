import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import cv2


SRC_ROOT = Path(__file__).resolve().parents[1]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rear_dog_prediction.common import (
    DEFAULT_MODEL_PATH,
    OUTPUT_ROOT,
    RearDogSelector,
    build_frame_record,
    calculate_motion_vector,
    extract_tracked_candidates,
    load_yolo_model,
    resolve_project_path,
    select_main_flock,
)


def build_rear_dog_tracking_dataset(
    video_paths: list[Path],
    model_path: Path,
    confidence: float,
    image_size: int,
    motion_window: int,
    motion_dead_zone: float,
    rear_projection_threshold: float,
    lateral_penalty: float,
    run_name: str,
) -> None:
    resolved_videos = [
        resolve_project_path(video_path)
        for video_path in video_paths
    ]

    for video_path in resolved_videos:
        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

    model_path = resolve_project_path(
        model_path
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    output_directory = OUTPUT_ROOT / run_name
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_csv = output_directory / "rear_dog_tracking_dataset.csv"
    model = load_yolo_model(model_path)

    print("=" * 70)
    print("BUILDING REAR DOG TRACKING DATASET")
    print("=" * 70)
    print(f"Videos: {len(resolved_videos)}")
    print(f"Model: {model_path}")
    print(f"Output: {output_csv}")
    print()

    total_rows = 0
    total_frames = 0

    with output_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "video_id",
                "video_path",
                "frame",
                "time_seconds",
                "frame_width",
                "frame_height",
                "segment_id",
                "flock_track_id",
                "dog_track_id",
                "flock_center_x",
                "flock_center_y",
                "dog_center_x",
                "dog_center_y",
                "flock_box_x1",
                "flock_box_y1",
                "flock_box_x2",
                "flock_box_y2",
                "dog_box_x1",
                "dog_box_y1",
                "dog_box_x2",
                "dog_box_y2",
                "flock_width",
                "flock_height",
                "dog_width",
                "dog_height",
                "flock_confidence",
                "dog_confidence",
                "rear_projection",
                "lateral_distance",
                "selection_score",
            ],
        )

        writer.writeheader()

        for video_index, video_path in enumerate(resolved_videos):
            capture = cv2.VideoCapture(
                str(video_path)
            )

            if not capture.isOpened():
                raise RuntimeError(
                    f"Cannot open video: {video_path}"
                )

            fps = capture.get(
                cv2.CAP_PROP_FPS
            )
            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0:
                capture.release()
                raise RuntimeError(
                    f"Invalid FPS in {video_path}"
                )

            video_id = f"video_{video_index:02d}_{video_path.stem}"
            active_flock_track_id = None
            last_rear_direction = None
            flock_trajectory: deque[tuple[int, int]] = deque(
                maxlen=max(2 * motion_window, 60),
            )
            selector = RearDogSelector(
                min_projection=rear_projection_threshold,
                lateral_penalty=lateral_penalty,
            )

            frame_index = 0
            current_segment_id = 0
            previous_visible_track_id = None

            print(f"Processing {video_id}: {video_path.name}")

            while True:
                success, frame = capture.read()

                if not success:
                    break

                result = model.track(
                    source=frame,
                    persist=True,
                    tracker="botsort.yaml",
                    classes=[0, 1],
                    conf=confidence,
                    iou=0.5,
                    imgsz=image_size,
                    verbose=False,
                )[0]

                flock_candidates, dog_candidates = extract_tracked_candidates(
                    result
                )

                selected_flock = select_main_flock(
                    candidates=flock_candidates,
                    active_track_id=active_flock_track_id,
                )

                if selected_flock is not None:
                    active_flock_track_id = selected_flock["track_id"]
                    flock_trajectory.append(
                        selected_flock["center"]
                    )

                    motion_direction, _ = calculate_motion_vector(
                        trajectory=flock_trajectory,
                        window=motion_window,
                        dead_zone=motion_dead_zone,
                    )

                    if motion_direction is not None:
                        last_rear_direction = -motion_direction

                    selected_dog = selector.select(
                        dog_candidates=dog_candidates,
                        flock_center=selected_flock["center"],
                        rear_direction=last_rear_direction,
                    )

                    if selected_dog is not None:
                        current_track_id = selected_dog["track_id"]

                        if current_track_id != previous_visible_track_id:
                            current_segment_id += 1

                        record = build_frame_record(
                            video_id=video_id,
                            video_path=video_path,
                            frame_index=frame_index,
                            fps=fps,
                            frame_width=frame_width,
                            frame_height=frame_height,
                            flock_candidate=selected_flock,
                            dog_candidate=selected_dog,
                            segment_id=current_segment_id,
                        )

                        writer.writerow(record)
                        total_rows += 1
                        previous_visible_track_id = current_track_id

                    else:
                        selector.clear_active_track()
                        previous_visible_track_id = None

                else:
                    selector.clear_active_track()
                    previous_visible_track_id = None

                frame_index += 1
                total_frames += 1

                if frame_index % 150 == 0:
                    print(
                        f"  {frame_index}/{frame_count} frames"
                    )

            capture.release()

    print()
    print("=" * 70)
    print("REAR DOG DATASET READY")
    print("=" * 70)
    print(f"Frames processed: {total_frames}")
    print(f"Tracked rows: {total_rows}")
    print(f"CSV: {output_csv}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a multi-video temporal dataset for rear dog prediction."
        )
    )

    parser.add_argument(
        "--videos",
        type=Path,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=960,
    )

    parser.add_argument(
        "--motion-window",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--motion-dead-zone",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--rear-projection-threshold",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--lateral-penalty",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--name",
        type=str,
        default="rear_dog_dataset",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    build_rear_dog_tracking_dataset(
        video_paths=args.videos,
        model_path=args.model,
        confidence=args.confidence,
        image_size=args.image_size,
        motion_window=args.motion_window,
        motion_dead_zone=args.motion_dead_zone,
        rear_projection_threshold=args.rear_projection_threshold,
        lateral_penalty=args.lateral_penalty,
        run_name=args.name,
    )


if __name__ == "__main__":
    main()
