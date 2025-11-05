from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import mujoco as mj
import numpy as np


@dataclass
class BodyPose:
    pos: np.ndarray  # shape (3, )
    rot: np.ndarray  # quaternion wxyz, shape (4, )


class RobotKinematics:
    """Lightweight MuJoCo-based kinematics helper.

    This utility mirrors the role of :class:`KinematicsModel` used in the forward
    retargeting pipeline, but is tailored for converting robot joint states back
    to human motion.  It exposes helpers for composing full ``qpos`` vectors and
    retrieving world-frame poses for all bodies in the model.
    """

    def __init__(self, xml_path: Path | str) -> None:
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"Robot MJCF xml not found: {self.xml_path}")

        self.model = mj.MjModel.from_xml_path(str(self.xml_path))
        self.data = mj.MjData(self.model)

        self._body_names: List[str] = []
        for body_id in range(self.model.nbody):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body_id)
            self._body_names.append(name if name is not None else "")

    @property
    def nq(self) -> int:
        return self.model.nq

    def compose_qpos(
        self,
        root_pos: Sequence[float],
        root_rot_wxyz: Sequence[float],
        dof_pos: Sequence[float],
    ) -> np.ndarray:
        """Compose a full MuJoCo ``qpos`` vector from root pose + actuated DoFs."""

        qpos = np.zeros(self.nq, dtype=np.float64)
        qpos[:3] = np.asarray(root_pos, dtype=np.float64)
        qpos[3:7] = np.asarray(root_rot_wxyz, dtype=np.float64)
        remaining = self.nq - 7
        dof_array = np.asarray(dof_pos, dtype=np.float64)
        if dof_array.size != remaining:
            raise ValueError(
                f"Unexpected DoF dimension: expected {remaining}, got {dof_array.size}"
            )
        qpos[7:] = dof_array
        return qpos

    def compose_qpos_sequence(
        self,
        root_pos: np.ndarray,
        root_rot_wxyz: np.ndarray,
        dof_pos: np.ndarray,
    ) -> np.ndarray:
        """Stack ``qpos`` vectors for a motion sequence."""

        num_frames = root_pos.shape[0]
        qpos_seq = np.zeros((num_frames, self.nq), dtype=np.float64)
        for idx in range(num_frames):
            qpos_seq[idx] = self.compose_qpos(root_pos[idx], root_rot_wxyz[idx], dof_pos[idx])
        return qpos_seq

    def forward_kinematics(self, qpos: np.ndarray) -> Dict[str, BodyPose]:
        """Compute world-frame pose for every body in the model."""

        if qpos.shape[-1] != self.nq:
            raise ValueError(f"qpos dimension mismatch: expected {self.nq}, got {qpos.shape[-1]}")

        self.data.qpos[:] = qpos
        mj.mj_forward(self.model, self.data)

        poses: Dict[str, BodyPose] = {}
        for body_id, body_name in enumerate(self._body_names):
            if not body_name:
                continue
            pos = self.data.xpos[body_id].copy()
            quat = self.data.xquat[body_id].copy()  # wxyz
            poses[body_name] = BodyPose(pos=pos, rot=quat)
        return poses

    def forward_kinematics_sequence(self, qpos_sequence: np.ndarray) -> List[Dict[str, BodyPose]]:
        """Run forward kinematics for a batch of ``qpos`` vectors."""

        poses: List[Dict[str, BodyPose]] = []
        for frame_qpos in qpos_sequence:
            poses.append(self.forward_kinematics(frame_qpos))
        return poses


__all__ = ["BodyPose", "RobotKinematics"]

