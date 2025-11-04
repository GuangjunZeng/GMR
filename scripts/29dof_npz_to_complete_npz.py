#!/usr/bin/env python3

"""Convert 29-DoF robot NPZ sequences (LaFAN1-style) into SMPLX-style NPZ files.

This utility inverts the retargeting pipeline by taking a single NPZ file
produced from robot retargeting (typically generated via
``29dof_pkl_to_npz_multiple.py``) and reconstructing the SMPLX parameters so
that the output matches the structure expected by ``smplx_to_robot_batch.py``
(e.g. ``server3_data/locomotion/reference/000002.npz``).

The resulting NPZ can then be fed into ``robot_to_smplx_batch.py``.
"""



from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Iterable, List, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


# Example usage:
# python scripts/29dof_npz_to_complete_npz.py  --input /home/retarget/workbench/server3_data/locomotion/robot/ik_based/npz/000001.npz  --output-dir /home/retarget/workbench/server3_data/locomotion/human/ik_based/npz


# Ensure we can import helper utilities that live in the same scripts/ folder
HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

try:
    from robot_to_smplx_batch import (  # type: ignore
        RobotToSMPLXBatchConverter,
        parse_betas,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
    raise ModuleNotFoundError(
        "Unable to import robot_to_smplx_batch utilities. "
        "Please ensure this script resides next to robot_to_smplx_batch.py "
        "inside the GMR/scripts directory."
    ) from exc


EXPECTED_JOINT_NAMES: Tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


DEFAULT_ROBOT = "unitree_g1"

DEFAULT_BETAS = np.array(
    [
        0.63490343,
        0.22382046,
        -1.02493083,
        0.44071582,
        -0.99539453,
        -2.14731956,
        1.5268985,
        -0.18637267,
        2.42483139,
        1.88858294,
    ],
    dtype=np.float64,
)


def _as_numpy(array: np.ndarray, target_shape: Tuple[int, ...] | None = None) -> np.ndarray:
    """Cast to float64 numpy array and optionally validate its shape."""

    result = np.asarray(array, dtype=np.float64)
    if target_shape is not None and result.shape != target_shape:
        raise ValueError(f"Expected shape {target_shape}, got {result.shape}")
    return result


def _infer_include_base(full_data: np.ndarray, joint_count: int, flag: np.ndarray | None) -> bool:
    if flag is not None:
        value = bool(np.array(flag).item())
        return value
    return full_data.shape[1] == joint_count + 7


def _read_joint_names(raw: np.ndarray | Iterable[str] | None) -> List[str]:
    if raw is None:
        raise ValueError("Input NPZ must contain 'joint_names' to validate LaFAN1 structure.")
    array = np.asarray(list(raw))
    return [str(x) for x in array]


def _validate_joint_names(joint_names: List[str]) -> None:
    if len(joint_names) != len(EXPECTED_JOINT_NAMES):
        raise ValueError(
            f"Joint count mismatch: expected {len(EXPECTED_JOINT_NAMES)}, got {len(joint_names)}"
        )
    for expected, actual in zip(EXPECTED_JOINT_NAMES, joint_names):
        if expected != actual:
            raise ValueError(
                "Input NPZ joint order does not match the standard LaFAN1 29-DoF layout. "
                f"Mismatch at joint '{actual}' (expected '{expected}')."
            )


def load_29dof_npz(npz_path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load robot NPZ and extract root pose plus joint DoFs."""

    if not npz_path.exists():
        raise FileNotFoundError(f"Input NPZ file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    joint_names = _read_joint_names(data.get("joint_names"))
    _validate_joint_names(joint_names)

    if "full_data" in data.files:
        full_data = _as_numpy(data["full_data"])
    elif "joints" in data.files:
        full_data = _as_numpy(data["joints"])
    else:
        raise ValueError(
            "Input NPZ must contain either 'full_data' or 'joints' array to describe joint motions."
        )

    num_frames = full_data.shape[0]
    include_base = _infer_include_base(full_data, len(joint_names), data.get("include_base"))

    if include_base:
        if "root_pos" in data.files:
            root_pos = _as_numpy(data["root_pos"])
        else:
            root_pos = full_data[:, :3]

        if root_pos.ndim == 1:
            root_pos = np.repeat(root_pos[None, :], num_frames, axis=0)

        if "root_quat" in data.files:
            root_quat_xyzw = _as_numpy(data["root_quat"])
        elif "root_rot" in data.files:
            root_quat_xyzw = _as_numpy(data["root_rot"])
        else:
            root_quat_xyzw = full_data[:, 3:7]

        if root_quat_xyzw.ndim == 1:
            root_quat_xyzw = np.repeat(root_quat_xyzw[None, :], num_frames, axis=0)

        dof_pos = full_data[:, -len(joint_names) :]
    else:
        root_pos = np.zeros((num_frames, 3), dtype=np.float64)
        root_quat_xyzw = np.zeros((num_frames, 4), dtype=np.float64)
        root_quat_xyzw[:, 3] = 1.0
        dof_pos = full_data

    fps = None
    for key in ("fps", "motion_fps", "frame_rate", "mocap_frame_rate"):
        if key in data.files:
            fps = float(np.array(data[key]).item())
            break

    return (
        _as_numpy(root_pos),
        _as_numpy(root_quat_xyzw),
        _as_numpy(dof_pos),
        fps if fps is not None else float("nan"),
    )


def build_qpos(root_pos: np.ndarray, root_quat_xyzw: np.ndarray, dof_pos: np.ndarray) -> np.ndarray:
    if not (root_pos.shape[0] == root_quat_xyzw.shape[0] == dof_pos.shape[0]):
        raise ValueError("root_pos, root_quat, and dof_pos must have the same number of frames")
    # warning： 需要调整顺序吗
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    return np.concatenate([root_pos, root_quat_wxyz, dof_pos], axis=1)


def convert_single_npz(
    npz_path: pathlib.Path,
    output_path: pathlib.Path,
    smplx_folder: pathlib.Path,
    gender: str,
    betas: np.ndarray,
    fps: float,
    show_progress: bool = False,
) -> Tuple[int, float]:
    root_pos, root_quat_xyzw, dof_pos, inferred_fps = load_29dof_npz(npz_path)

    if np.isnan(inferred_fps) and np.isnan(fps):
        raise ValueError(
            "FPS is not present in the NPZ and no override was provided via --fps."
        )

    motion_fps = fps if not np.isnan(fps) else inferred_fps

    qpos_matrix = build_qpos(root_pos, root_quat_xyzw, dof_pos)

    converter = RobotToSMPLXBatchConverter(
        robot_type=DEFAULT_ROBOT,
        smplx_model_path=str(smplx_folder),
        gender=gender,
    )

    frame_indices = range(qpos_matrix.shape[0])
    if show_progress:
        from tqdm import tqdm

        frame_indices = tqdm(frame_indices, desc="Converting frames", leave=False)

    smplx_joints_list = []
    for idx in frame_indices:
        qpos = qpos_matrix[idx]
        smplx_joints = converter.robot_frame_to_smplx_joints(qpos)
        smplx_joints_list.append(smplx_joints)

    if betas is None or np.asarray(betas).size == 0:
        betas_array = DEFAULT_BETAS.astype(np.float64, copy=True)
    else:
        betas_array = np.asarray(betas, dtype=np.float64)

    smplx_params = converter._joints_to_smplx_params(smplx_joints_list, betas=betas_array)

    # Shapes
    num_frames = smplx_params["pose_body"].shape[0]

    # Prepare root orientation quaternions (xyzw)
    root_quat_xyzw = R.from_rotvec(smplx_params["root_orient"]).as_quat()

    # Build joints_local via SMPL-X forward pass (full_pose -> (T, 55, 3))
    with torch.no_grad():
        betas_t = torch.tensor(smplx_params["betas"], dtype=torch.float32).unsqueeze(0).repeat(num_frames, 1)
        root_orient_t = torch.tensor(smplx_params["root_orient"], dtype=torch.float32)
        body_pose_t = torch.tensor(smplx_params["pose_body"], dtype=torch.float32)
        trans_t = torch.tensor(smplx_params["trans"], dtype=torch.float32)
        left_hand_pose_t = torch.zeros((num_frames, 45), dtype=torch.float32)
        right_hand_pose_t = torch.zeros((num_frames, 45), dtype=torch.float32)
        jaw_pose_t = torch.zeros((num_frames, 3), dtype=torch.float32)
        leye_pose_t = torch.zeros((num_frames, 3), dtype=torch.float32)
        reye_pose_t = torch.zeros((num_frames, 3), dtype=torch.float32)
        expression_size = getattr(converter.smplx_model, "num_expression_coeffs", 10)
        expression_t = torch.zeros((num_frames, expression_size), dtype=torch.float32)

        smplx_output = converter.smplx_model(
            betas=betas_t,
            global_orient=root_orient_t,
            body_pose=body_pose_t,
            transl=trans_t,
            left_hand_pose=left_hand_pose_t,
            right_hand_pose=right_hand_pose_t,
            jaw_pose=jaw_pose_t,
            leye_pose=leye_pose_t,
            reye_pose=reye_pose_t,
            expression=expression_t,
            return_full_pose=True,
        )
        full_pose = smplx_output.full_pose.reshape(num_frames, -1, 3).detach().cpu().numpy()

    # Compose output in the exact reference structure and dtypes
    out = {
        # scalars/metadata
        "gender": np.array(gender),
        "betas": smplx_params["betas"].astype(np.float32),
        # body/hand axis-angle
        "pose_body": smplx_params["pose_body"].astype(np.float32),
        "pose_hand": np.zeros((num_frames, 90), dtype=np.float32),
        # root (smpl_) pose, also duplicate as pelvis_*
        "smpl_trans": smplx_params["trans"].astype(np.float32),
        "smpl_quat_xyzw": root_quat_xyzw.astype(np.float32),
        "pelvis_trans": smplx_params["trans"].astype(np.float32),
        "pelvis_quat_xyzw": root_quat_xyzw.astype(np.float64),
        # full local axis-angle for all joints
        "joints_local": full_pose.astype(np.float64),
        # fps as integer scalar
        "fps": np.array(int(round(motion_fps)), dtype=np.int64),
    }

    np.savez(output_path, **out)

    return qpos_matrix.shape[0], motion_fps


def default_output_path(input_path: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
    stem = input_path.stem
    return output_dir / f"{stem}.npz"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reverse retarget a single 29-DoF robot NPZ back to SMPLX-style NPZ."
    )

    parser.add_argument("--input", required=True, help="Path to the 29-DoF robot NPZ file")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output NPZ path. Overrides --output-dir if provided.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to place the converted NPZ (defaults to server3_data/locomotion/human/ik_based/npz)",
    )
    parser.add_argument(
        "--gender",
        choices=["neutral", "female", "male"],
        default="female",
        help="SMPL-X gender variant",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="*",
        default=None,
        help="Optional SMPL-X shape coefficients (defaults to reference template)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Override FPS (defaults to 30 if input NPZ lacks FPS)",
    )
    parser.add_argument(
        "--smplx-folder",
        default=None,
        help="Directory containing SMPL-X body models (defaults to scripts/assets/body_models)",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Display per-frame progress bar",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite the output file if it already exists (default: True)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    input_path = pathlib.Path(args.input).expanduser().resolve()
    if args.output:
        output_path = pathlib.Path(args.output).expanduser().resolve()
        output_dir = output_path.parent
    else:
        if args.output_dir:
            output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
        else:
            output_dir = (PROJECT_ROOT / "server3_data" / "locomotion" / "human" / "ik_based" / "npz").resolve()
        output_path = default_output_path(input_path, output_dir)

    smplx_folder = (
        pathlib.Path(args.smplx_folder).expanduser().resolve()
        if args.smplx_folder
        else HERE / "assets" / "body_models"
    )

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file {output_path} already exists. Use --overwrite to replace it."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    betas = parse_betas(args.betas)

    frame_count, motion_fps = convert_single_npz(
        npz_path=input_path,
        output_path=output_path,
        smplx_folder=smplx_folder,
        gender=args.gender,
        betas=betas,
        fps=args.fps,
        show_progress=args.show_progress,
    )

    print(
        f"✅ Converted {input_path.name}: frames={frame_count}, fps={motion_fps:.2f} -> {output_path}"
    )


if __name__ == "__main__":
    main()


