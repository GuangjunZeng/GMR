
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pickle


def _load_pickle_motion(path: Path) -> Tuple[Dict[str, Any], float, np.ndarray, np.ndarray, np.ndarray, Any, Any]:
    with path.open("rb") as f:
        motion_data = pickle.load(f)

    motion_fps = float(motion_data["fps"])
    motion_root_pos = np.asarray(motion_data["root_pos"], dtype=np.float64)
    motion_root_rot_xyzw = np.asarray(motion_data["root_rot"], dtype=np.float64)
    motion_root_rot = motion_root_rot_xyzw[:, [3, 0, 1, 2]]  # xyzw -> wxyz
    motion_dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float64)
    motion_local_body_pos = motion_data.get("local_body_pos")
    motion_link_body_list = motion_data.get("link_body_list")

    return (
        motion_data,
        motion_fps,
        motion_root_pos,
        motion_root_rot,
        motion_dof_pos,
        motion_local_body_pos,
        motion_link_body_list,
    )


def _load_npz_motion(path: Path) -> Tuple[Dict[str, Any], float, np.ndarray, np.ndarray, np.ndarray, Any, Any]:
    with np.load(path, allow_pickle=True) as data:
        if "qpos" in data:
            qpos = np.asarray(data["qpos"], dtype=np.float64)
        elif "motion" in data:
            qpos = np.asarray(data["motion"], dtype=np.float64)
        else:
            raise KeyError(
                f"NPZ motion file {path} must contain 'qpos' (xyz + quat + dof) array"
            )

        if qpos.ndim != 2 or qpos.shape[1] < 7:
            raise ValueError(
                f"Unexpected qpos shape {qpos.shape} in {path}; expected (frames, 7 + dofs)"
            )

        root_pos = qpos[:, :3]
        root_rot_xyzw = qpos[:, 3:7]
        root_rot = np.concatenate([root_rot_xyzw[:, 3:4], root_rot_xyzw[:, :3]], axis=1)
        dof_pos = qpos[:, 7:]

        fps = None
        for key in ("fps", "frame_rate", "framerate", "mocap_frame_rate"):
            if key in data:
                fps_array = np.asarray(data[key])
                fps = float(fps_array.item() if fps_array.shape == () else fps_array.flatten()[0])
                break
        if fps is None:
            fps = 30.0

        motion_data: Dict[str, Any] = {
            "fps": fps,
            "root_pos": root_pos,
            "root_rot_xyzw": root_rot_xyzw,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "qpos": qpos,
        }

        if "local_body_pos" in data:
            motion_data["local_body_pos"] = data["local_body_pos"]
        if "link_body_list" in data:
            link_array = data["link_body_list"]
            motion_data["link_body_list"] = (
                link_array.tolist() if isinstance(link_array, np.ndarray) else link_array
            )

        motion_local_body_pos = motion_data.get("local_body_pos")
        motion_link_body_list = motion_data.get("link_body_list")

    return (
        motion_data,
        fps,
        root_pos,
        root_rot,
        dof_pos,
        motion_local_body_pos,
        motion_link_body_list,
    )


def load_robot_motion(motion_file: str):
    """Load robot motion data from either pickle or NPZ format."""

    path = Path(motion_file)
    if not path.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_file}")

    suffix = path.suffix.lower()
    if suffix == ".pkl":
        return _load_pickle_motion(path)
    if suffix == ".npz":
        return _load_npz_motion(path)

    raise ValueError(f"Unsupported motion file format '{suffix}' for {motion_file}")


