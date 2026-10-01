# utils/video.py

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2


def get_video_frame_count(video_path: str) -> int:
    """
    Get total frame count of a video.

    Args:
        video_path: Path to video file.

    Returns:
        Total number of frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if frame_count <= 0:
        raise ValueError(f"Invalid frame count for video: {video_path}")

    return frame_count


def sample_frame_indices(
    num_frames: int,
    num_samples: int,
    include_first: bool = True,
    include_last: bool = True,
) -> List[int]:
    """
    Uniformly sample frame indices from a video.

    Args:
        num_frames: Total number of frames in the video.
        num_samples: Number of frames to sample.
        include_first: Whether to force include the first frame.
        include_last: Whether to force include the last frame.

    Returns:
        A sorted list of unique frame indices (0-based).
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    if num_samples >= num_frames:
        return list(range(num_frames))

    if num_samples == 1:
        return [0]

    if include_first and include_last and num_samples >= 2:
        step = (num_frames - 1) / (num_samples - 1)
        indices = [round(i * step) for i in range(num_samples)]
    else:
        step = num_frames / num_samples
        indices = [round((i + 0.5) * step) for i in range(num_samples)]
        indices = [min(max(idx, 0), num_frames - 1) for idx in indices]

    # ensure uniqueness and sorted order
    indices = sorted(set(indices))

    # if rounding caused fewer samples, fill missing indices
    candidate = 0
    while len(indices) < num_samples:
        if candidate not in indices:
            indices.append(candidate)
        candidate += 1
        if candidate >= num_frames:
            break

    indices = sorted(indices)[:num_samples]
    return indices


def extract_frames_by_indices(
    video_path: str,
    frame_indices: List[int],
    output_dir: str,
    prefix: str = "frame",
) -> List[str]:
    """
    Extract specified frame indices and save them as images.

    Args:
        video_path: Path to input video.
        frame_indices: List of 0-based frame indices.
        output_dir: Directory to save extracted frames.
        prefix: Filename prefix.

    Returns:
        List of saved frame paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    wanted = sorted(set(frame_indices))
    saved_paths: List[str] = []
    wanted_set = set(wanted)

    current_idx = 0
    save_counter = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if current_idx in wanted_set:
            # OpenCV reads BGR; save directly with cv2.imwrite
            save_name = f"{prefix}_{save_counter:04d}.png"
            save_path = output_path / save_name
            cv2.imwrite(str(save_path), frame)
            saved_paths.append(str(save_path).replace("\\", "/"))
            save_counter += 1

            if save_counter >= len(wanted):
                break

        current_idx += 1

    cap.release()

    if len(saved_paths) != len(wanted):
        raise RuntimeError(
            f"Expected to save {len(wanted)} frames, but saved {len(saved_paths)} "
            f"from video: {video_path}"
        )

    return saved_paths


def extract_uniform_frames(
    video_path: str,
    output_dir: str,
    num_samples: int = 6,
    prefix: str = "frame",
    include_first: bool = True,
    include_last: bool = True,
) -> List[str]:
    """
    Uniformly sample frames from a video and save them.

    Args:
        video_path: Path to input video.
        output_dir: Directory to save extracted frames.
        num_samples: Number of frames to sample.
        prefix: Filename prefix.
        include_first: Whether to force include first frame.
        include_last: Whether to force include last frame.

    Returns:
        List of saved frame paths.
    """
    num_frames = get_video_frame_count(video_path)
    indices = sample_frame_indices(
        num_frames=num_frames,
        num_samples=num_samples,
        include_first=include_first,
        include_last=include_last,
    )
    return extract_frames_by_indices(
        video_path=video_path,
        frame_indices=indices,
        output_dir=output_dir,
        prefix=prefix,
    )