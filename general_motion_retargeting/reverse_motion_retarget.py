from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import smplx
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES

from .params import REVERSE_IK_CONFIG_DICT, ROBOT_XML_DICT
from .robot import BodyPose, RobotKinematics


class RobotToSMPLXRetargeting:
    """Framework-style wrapper mirroring :class:`GeneralMotionRetargeting` for reverse conversion."""

    def __init__(
        self,
        robot_type: str,
        smplx_model_path: Path | str,
        gender: str = "neutral",
        ik_config_path: Optional[Path | str] = None,
    ) -> None:
        # ===== Robot kinematics assets =====
        self.robot_type = robot_type
        self.robot_xml_path = ROBOT_XML_DICT[robot_type]
        self.robot_kinematics = RobotKinematics(self.robot_xml_path)

        # ===== IK configuration =====
        if ik_config_path is None:
            ik_config_path = REVERSE_IK_CONFIG_DICT.get(robot_type)
        if ik_config_path is None:
            raise ValueError(f"No reverse IK config registered for robot type: {robot_type}")
        self.ik_config_path = Path(ik_config_path)
        if not self.ik_config_path.exists():
            raise FileNotFoundError(f"IK config not found: {self.ik_config_path}")
        self.ik_config = self._load_ik_config(self.ik_config_path)

        self.robot_root_name = self.ik_config["robot_root_name"]
        self.smplx_root_name = self.ik_config["human_root_name"]
        self.use_ik_match_table1 = self.ik_config.get("use_ik_match_table1", True)
        self.use_ik_match_table2 = self.ik_config.get("use_ik_match_table2", True)
        self.robot_scale_table = self.ik_config.get("robot_scale_table", {})
        self.ground_height = float(self.ik_config.get("ground_height", 0.0))

        self.ground_offset = 0.0
        self._ground_offset_initialized = False

        # ===== SMPL-X body model =====
        self.smplx_model_path = Path(smplx_model_path)
        if not self.smplx_model_path.exists():
            raise FileNotFoundError(f"SMPL-X model folder not found: {self.smplx_model_path}")

        self.gender = gender
        self.smplx_model = smplx.create(
            model_path=str(self.smplx_model_path),
            model_type="smplx",
            gender=gender,
            use_pca=False,
        )

        self.smplx_joint_names = JOINT_NAMES[: len(self.smplx_model.parents)]
        self.smplx_name_to_idx = {name: i for i, name in enumerate(self.smplx_joint_names)}
        self.smplx_parents = self.smplx_model.parents.detach().cpu().numpy().astype(int)

        self.num_betas = getattr(self.smplx_model, "num_betas", None)
        if self.num_betas is None:
            betas_attr = getattr(self.smplx_model, "betas", None)
            if betas_attr is not None and hasattr(betas_attr, "shape") and betas_attr.shape[-1] > 0:
                self.num_betas = int(betas_attr.shape[-1])
        if self.num_betas is None:
            self.num_betas = 10

        self.setup_retarget_configuration()

    # ---------------------------------------------------------------------
    # Configuration helpers
    # ---------------------------------------------------------------------
    def _load_ik_config(self, config_path: Path) -> Dict:
        with config_path.open("r") as f:
            return json.load(f)

    def setup_retarget_configuration(self) -> None:
        self.smplx_joint_to_robot: Dict[str, str] = {}
        self.pos_offsets: Dict[str, np.ndarray] = {}
        self.rot_offsets: Dict[str, np.ndarray] = {}
        self.pos_weights: Dict[str, float] = {}
        self.rot_weights: Dict[str, float] = {}

        def register_entry(entry_key: str, entry_value: List, table_name: str) -> None:
            if not entry_value:
                return

            if entry_key in JOINT_NAMES:
                smplx_joint = entry_key
                robot_body = entry_value[0]
            else:
                robot_body = entry_key
                smplx_joint = entry_value[0]

            if smplx_joint not in JOINT_NAMES:
                return

            pos_weight = entry_value[1]
            rot_weight = entry_value[2]
            pos_offset = np.asarray(entry_value[3], dtype=np.float64)
            rot_offset = np.asarray(entry_value[4], dtype=np.float64)

            self.smplx_joint_to_robot[smplx_joint] = robot_body
            self.pos_offsets[smplx_joint] = pos_offset
            self.rot_offsets[smplx_joint] = rot_offset
            self.pos_weights[smplx_joint] = pos_weight
            self.rot_weights[smplx_joint] = rot_weight

        for table_name in ("ik_match_table1", "ik_match_table2"):
            table = self.ik_config.get(table_name, {})
            for key, value in table.items():
                register_entry(key, value, table_name)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def motion_to_smplx_params(
        self,
        root_pos: np.ndarray,
        root_rot_wxyz: np.ndarray,
        dof_pos: np.ndarray,
        betas: Optional[np.ndarray] = None,
        show_progress: bool = False,
    ) -> Dict[str, np.ndarray]:
        qpos_sequence = self.compose_robot_motion_sequence(root_pos, root_rot_wxyz, dof_pos)
        frames = self.map_robot_motion(qpos_sequence, show_progress=show_progress)
        return self.frames_to_smplx_parameters(frames, betas)

    # ---------------------------------------------------------------------
    # Core pipeline steps
    # ---------------------------------------------------------------------
    def compose_robot_motion_sequence(
        self,
        root_pos: np.ndarray,
        root_rot_wxyz: np.ndarray,
        dof_pos: np.ndarray,
    ) -> np.ndarray:
        return self.robot_kinematics.compose_qpos_sequence(root_pos, root_rot_wxyz, dof_pos)

    def map_robot_motion(
        self,
        qpos_sequence: np.ndarray,
        show_progress: bool = False,
    ) -> List[Dict[str, Dict[str, np.ndarray]]]:
        iterator = range(len(qpos_sequence))
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Processing frames", leave=False)

        frames: List[Dict[str, Dict[str, np.ndarray]]] = []
        for idx in iterator:
            robot_bodies = self.robot_kinematics.forward_kinematics(qpos_sequence[idx])
            targets = self.update_targets(robot_bodies)
            joints = self.retarget(targets)
            frames.append(joints)
        return frames

    def frames_to_smplx_parameters(
        self,
        smplx_frames: List[Dict[str, Dict[str, np.ndarray]]],
        betas: Optional[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        betas_array = self.prepare_betas(betas)
        root_orient_list: List[np.ndarray] = []
        trans_list: List[np.ndarray] = []
        body_pose_list: List[np.ndarray] = []

        for frame_joints in smplx_frames:
            if self.smplx_root_name in frame_joints:
                root_pos = frame_joints[self.smplx_root_name]["pos"]
                root_rot = frame_joints[self.smplx_root_name]["rot"]
                trans_list.append(root_pos)
                root_orient_list.append(R.from_quat(root_rot[[1, 2, 3, 0]]).as_rotvec())
            else:
                trans_list.append(np.zeros(3, dtype=np.float64))
                root_orient_list.append(np.zeros(3, dtype=np.float64))

            body_pose_list.append(self.compute_local_rotations(frame_joints))

        return {
            "betas": betas_array,
            "root_orient": np.asarray(root_orient_list, dtype=np.float64),
            "trans": np.asarray(trans_list, dtype=np.float64),
            "pose_body": np.asarray(body_pose_list, dtype=np.float64),
        }

    # ---------------------------------------------------------------------
    # Frame-level helpers
    # ---------------------------------------------------------------------
    def update_targets(self, robot_data: Dict[str, BodyPose], offset_to_ground: bool = True) -> Dict[str, BodyPose]:
        robot_data = self.to_numpy(robot_data)
        robot_data = self.scale_robot_data(robot_data)
        robot_data = self.offset_robot_data(robot_data)

        if offset_to_ground:
            if not self._ground_offset_initialized:
                try:
                    lowest_z = min(pose.pos[2] for pose in robot_data.values())
                    desired = self.ground_height
                    self.set_ground_offset(max(0.0, desired - lowest_z))
                except ValueError:
                    self.set_ground_offset(0.0)
                self._ground_offset_initialized = True
            robot_data = self.apply_ground_offset(robot_data)

        self.scaled_robot_data = robot_data
        return robot_data

    def retarget(self, robot_data: Dict[str, BodyPose]) -> Dict[str, Dict[str, np.ndarray]]:
        smplx_targets: Dict[str, Dict[str, np.ndarray]] = {}
        for smplx_joint, robot_body in self.smplx_joint_to_robot.items():
            pose = robot_data.get(robot_body)
            if pose is None:
                continue
            smplx_pos, smplx_rot = self.apply_reverse_offsets(smplx_joint, pose)
            smplx_targets[smplx_joint] = {"pos": smplx_pos, "rot": smplx_rot}
        return smplx_targets

    def to_numpy(self, body_poses: Dict[str, BodyPose]) -> Dict[str, BodyPose]:
        return body_poses

    def scale_robot_data(self, robot_data: Dict[str, BodyPose]) -> Dict[str, BodyPose]:
        if not self.robot_scale_table:
            return robot_data

        root_pose = robot_data.get(self.robot_root_name)
        if root_pose is None:
            return robot_data

        scaled: Dict[str, BodyPose] = {}
        root_pos = root_pose.pos
        for body_name, pose in robot_data.items():
            scale = self.robot_scale_table.get(body_name)
            if scale is None or body_name == self.robot_root_name:
                scaled[body_name] = pose
                continue
            local = pose.pos - root_pos
            scaled_pos = local * scale + root_pos
            scaled[body_name] = BodyPose(pos=scaled_pos, rot=pose.rot)

        return scaled

    def offset_robot_data(self, robot_data: Dict[str, BodyPose]) -> Dict[str, BodyPose]:
        offset_data: Dict[str, BodyPose] = {}
        for smplx_joint, robot_body in self.smplx_joint_to_robot.items():
            pose = robot_data.get(robot_body)
            if pose is None:
                continue
            pos_offset = self.pos_offsets.get(smplx_joint, np.zeros(3))
            rot_offset = self.rot_offsets.get(smplx_joint)

            robot_R = R.from_quat(pose.rot[[1, 2, 3, 0]])
            if rot_offset is not None:
                offset_R = R.from_quat(rot_offset[[1, 2, 3, 0]])
                updated_rot = (robot_R * offset_R.inv()).as_quat(scalar_first=True)
            else:
                updated_rot = pose.rot

            updated_pos = pose.pos - robot_R.apply(pos_offset)
            offset_data[robot_body] = BodyPose(pos=updated_pos, rot=updated_rot)

        for body_name, pose in robot_data.items():
            offset_data.setdefault(body_name, pose)

        return offset_data

    def apply_ground_offset(self, body_poses: Dict[str, BodyPose]) -> Dict[str, BodyPose]:
        if self.ground_offset == 0.0:
            return body_poses
        shifted: Dict[str, BodyPose] = {}
        for body_name, pose in body_poses.items():
            shifted_pos = pose.pos + np.array([0.0, 0.0, self.ground_offset])
            shifted[body_name] = BodyPose(pos=shifted_pos, rot=pose.rot)
        return shifted

    def set_ground_offset(self, ground_offset: float) -> None:
        self.ground_offset = float(ground_offset)

    def apply_reverse_offsets(self, smplx_joint: str, pose: BodyPose) -> tuple[np.ndarray, np.ndarray]:
        return pose.pos, pose.rot

    # ---------------------------------------------------------------------
    # Parameter utilities
    # ---------------------------------------------------------------------
    def prepare_betas(self, betas: Optional[np.ndarray]) -> np.ndarray:
        if betas is None:
            return np.zeros(self.num_betas, dtype=np.float64)
        betas = np.asarray(betas, dtype=np.float64)
        if betas.shape[0] != self.num_betas:
            if betas.shape[0] < self.num_betas:
                betas = np.pad(betas, (0, self.num_betas - betas.shape[0]), mode="constant")
            else:
                betas = betas[: self.num_betas]
        return betas

    def compute_local_rotations(
        self,
        frame_joints: Dict[str, Dict[str, np.ndarray]],
    ) -> np.ndarray:
        body_pose = np.zeros(63, dtype=np.float64)
        global_rots: Dict[str, R] = {}

        for joint_name, joint_data in frame_joints.items():
            if joint_name not in self.smplx_name_to_idx:
                continue
            rot_quat = joint_data["rot"]
            global_rots[joint_name] = R.from_quat(rot_quat[[1, 2, 3, 0]])

        for i in range(1, min(22, len(self.smplx_joint_names))):
            joint_name = self.smplx_joint_names[i]
            parent_idx = self.smplx_parents[i]
            parent_name = self.smplx_joint_names[parent_idx]
            if joint_name not in global_rots or parent_name not in global_rots:
                continue
            parent_R = global_rots[parent_name]
            joint_R = global_rots[joint_name]
            local_R = parent_R.inv() * joint_R
            body_pose[(i - 1) * 3 : (i - 1) * 3 + 3] = local_R.as_rotvec()

        return body_pose


__all__ = ["RobotToSMPLXRetargeting"]

