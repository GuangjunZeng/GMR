#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reverse Retargeting: Robot (G1) to SMPL-X
将机器人动作数据反向转换为SMPL-X格式

注意：这是一个近似逆向过程，存在信息损失：
- G1的自由度少于SMPL-X
- 无法恢复手指精细动作、面部表情等
- betas（体型参数）需要手动指定或使用默认值
"""

import argparse
import pathlib
import pickle
import numpy as np
import torch
import smplx
import mujoco as mj
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES
import json
from rich import print
from tqdm import tqdm


class RobotToSMPLX:
    """机器人动作到SMPL-X的反向转换器"""
    
    def __init__(self, robot_type, robot_xml_path, ik_config_path, smplx_model_path, gender="neutral"):
        """
        初始化反向转换器
        
        Args:
            robot_type: 机器人类型名称
            robot_xml_path: 机器人MJCF XML文件路径
            ik_config_path: IK配置文件路径
            smplx_model_path: SMPL-X模型文件夹路径
            gender: SMPL-X性别 ("male", "female", "neutral")
        """
        self.robot_type = robot_type
        
        # 加载机器人模型
        print(f"📦 Loading robot model: {robot_xml_path}")
        self.robot_model = mj.MjModel.from_xml_path(str(robot_xml_path))
        self.robot_data = mj.MjData(self.robot_model)
        
        # 加载IK配置（用于反向映射）
        print(f"📦 Loading IK config: {ik_config_path}")
        with open(ik_config_path, 'r') as f:
            self.ik_config = json.load(f)
        
        # 构建反向映射表：SMPL-X关节名 -> 机器人body名
        self.smplx_to_robot_map = self._build_reverse_mapping()
        
        # 加载SMPL-X模型
        print(f"📦 Loading SMPL-X model: {smplx_model_path}")
        self.smplx_model = smplx.create(
            smplx_model_path,
            "smplx",
            gender=gender,
            use_pca=False,
        )
        
        # SMPL-X关节名称
        self.smplx_joint_names = JOINT_NAMES[:len(self.smplx_model.parents)]
        
        # 构建SMPL-X关节索引映射
        self.smplx_name_to_idx = {name: i for i, name in enumerate(self.smplx_joint_names)}
        
        print(f"✅ Reverse mapper initialized for {robot_type}")
        print(f"   Mapped {len(self.smplx_to_robot_map)} SMPL-X joints to robot bodies")
        
    def _build_reverse_mapping(self):
        """构建SMPL-X关节到机器人body的反向映射"""
        reverse_map = {}
        
        # 从ik_match_table1提取映射关系
        if "ik_match_table1" in self.ik_config:
            for robot_body_name, mapping_info in self.ik_config["ik_match_table1"].items():
                smplx_joint_name = mapping_info[0]  # 第一个元素是SMPL-X关节名
                pos_offset = np.array(mapping_info[2])  # 位置偏移
                rot_offset = np.array(mapping_info[3])  # 旋转偏移 (wxyz)
                
                reverse_map[smplx_joint_name] = {
                    'robot_body': robot_body_name,
                    'pos_offset': pos_offset,
                    'rot_offset': rot_offset
                }
        
        return reverse_map
    
    def _get_robot_body_pose(self, qpos):
        """
        从qpos获取机器人各个body的全局位置和方向
        
        Args:
            qpos: 机器人的完整状态 [root_pos(3), root_rot(4), dof_pos(N)]
            
        Returns:
            body_poses: dict {body_name: {'pos': np.array, 'rot': np.array(wxyz)}}
        """
        # 设置机器人状态
        self.robot_data.qpos[:] = qpos
        mj.mj_forward(self.robot_model, self.robot_data)
        
        # 提取所有body的位置和方向
        body_poses = {}
        for i in range(self.robot_model.nbody):
            body_name = mj.mj_id2name(self.robot_model, mj.mjtObj.mjOBJ_BODY, i)
            if body_name:
                # xpos: 全局位置
                pos = self.robot_data.xpos[i].copy()
                # xquat: 全局方向 (wxyz)
                quat = self.robot_data.xquat[i].copy()
                body_poses[body_name] = {'pos': pos, 'rot': quat}
        
        return body_poses
    
    def _apply_reverse_offset(self, robot_pos, robot_rot, pos_offset, rot_offset):
        """
        应用反向偏移从机器人坐标系转换到SMPL-X坐标系
        
        Args:
            robot_pos: 机器人body位置
            robot_rot: 机器人body方向 (wxyz quaternion)
            pos_offset: 位置偏移
            rot_offset: 旋转偏移 (wxyz quaternion)
            
        Returns:
            smplx_pos, smplx_rot: SMPL-X关节位置和方向
        """
        # 旋转偏移：去除robot到smplx的旋转
        robot_R = R.from_quat(robot_rot[[1,2,3,0]])  # wxyz -> xyzw
        offset_R = R.from_quat(rot_offset[[1,2,3,0]])  # wxyz -> xyzw
        smplx_R = robot_R * offset_R.inv()
        
        # 位置偏移：去除robot到smplx的位移（在robot坐标系下）
        smplx_pos = robot_pos - robot_R.apply(pos_offset)
        smplx_rot = smplx_R.as_quat(scalar_first=True)  # wxyz
        
        return smplx_pos, smplx_rot
    
    def robot_frame_to_smplx_joints(self, qpos):
        """
        将单帧机器人状态转换为SMPL-X关节信息
        
        Args:
            qpos: [root_pos(3), root_rot(4 wxyz), dof_pos(N)]
            
        Returns:
            smplx_joints: dict {joint_name: {'pos': np.array, 'rot': np.array(wxyz)}}
        """
        # 获取机器人body姿态
        body_poses = self._get_robot_body_pose(qpos)
        
        # 映射到SMPL-X关节
        smplx_joints = {}
        for smplx_joint_name, mapping in self.smplx_to_robot_map.items():
            robot_body_name = mapping['robot_body']
            pos_offset = mapping['pos_offset']
            rot_offset = mapping['rot_offset']
            
            if robot_body_name in body_poses:
                robot_pose = body_poses[robot_body_name]
                smplx_pos, smplx_rot = self._apply_reverse_offset(
                    robot_pose['pos'], 
                    robot_pose['rot'],
                    pos_offset,
                    rot_offset
                )
                smplx_joints[smplx_joint_name] = {
                    'pos': smplx_pos,
                    'rot': smplx_rot
                }
        
        return smplx_joints
    
    def _joints_to_smplx_params(self, smplx_joints_list, betas=None):
        """
        从关节列表拟合SMPL-X参数（使用简化方法）
        
        Args:
            smplx_joints_list: 每帧的SMPL-X关节字典列表
            betas: 体型参数 (16,)，如果为None则使用默认值
            
        Returns:
            smplx_params: dict containing pose_body, root_orient, trans
        """
        num_frames = len(smplx_joints_list)
        
        # 默认betas（标准体型）
        if betas is None:
            betas = np.zeros(16)
        
        # 初始化参数
        root_orient_list = []
        trans_list = []
        body_pose_list = []
        
        for frame_joints in tqdm(smplx_joints_list, desc="Converting to SMPL-X params"):
            # 提取pelvis（root）信息
            if 'pelvis' in frame_joints:
                pelvis_pos = frame_joints['pelvis']['pos']
                pelvis_rot = frame_joints['pelvis']['rot']  # wxyz
                
                # Root位置和方向
                trans_list.append(pelvis_pos)
                pelvis_R = R.from_quat(pelvis_rot[[1,2,3,0]])  # xyzw
                root_orient_list.append(pelvis_R.as_rotvec())
            else:
                # 使用默认值
                trans_list.append(np.zeros(3))
                root_orient_list.append(np.zeros(3))
            
            # 计算局部关节旋转（相对于父节点）
            body_pose = self._compute_local_rotations(frame_joints)
            body_pose_list.append(body_pose)
        
        smplx_params = {
            'betas': betas,
            'root_orient': np.array(root_orient_list),  # (N, 3)
            'trans': np.array(trans_list),  # (N, 3)
            'pose_body': np.array(body_pose_list),  # (N, 63)
        }
        
        return smplx_params
    
    def _compute_local_rotations(self, frame_joints):
        """
        计算局部关节旋转（相对于父节点）
        
        Args:
            frame_joints: 单帧的SMPL-X关节字典
            
        Returns:
            body_pose: (63,) 身体姿态参数
        """
        # SMPL-X body pose: 21个关节 x 3维旋转向量 = 63
        body_pose = np.zeros(63)
        
        # 获取父节点关系
        parents = self.smplx_model.parents.numpy()
        
        # 构建全局旋转矩阵字典
        global_rots = {}
        for joint_name, joint_data in frame_joints.items():
            if joint_name in self.smplx_name_to_idx:
                rot_quat = joint_data['rot']  # wxyz
                global_rots[joint_name] = R.from_quat(rot_quat[[1,2,3,0]])  # xyzw
        
        # 计算局部旋转（跳过pelvis，因为它是root_orient）
        # SMPL-X body joints: index 1-21 (pelvis is 0)
        for i in range(1, min(22, len(self.smplx_joint_names))):
            joint_name = self.smplx_joint_names[i]
            parent_idx = parents[i]
            parent_name = self.smplx_joint_names[parent_idx]
            
            if joint_name in global_rots and parent_name in global_rots:
                # 局部旋转 = 父节点的逆旋转 * 当前节点的全局旋转
                parent_R = global_rots[parent_name]
                joint_R = global_rots[joint_name]
                local_R = parent_R.inv() * joint_R
                
                # 存储为旋转向量
                body_pose[(i-1)*3:(i-1)*3+3] = local_R.as_rotvec()
            else:
                # 如果缺失，使用零旋转（中性姿态）
                body_pose[(i-1)*3:(i-1)*3+3] = np.zeros(3)
        
        return body_pose
    
    def convert_pkl_to_npz(self, pkl_path, output_npz_path, betas=None, fps=None):
        """
        将机器人PKL文件转换为SMPL-X NPZ文件
        
        Args:
            pkl_path: 输入的机器人pkl文件路径
            output_npz_path: 输出的SMPL-X npz文件路径
            betas: 体型参数，如果为None则使用默认
            fps: 帧率，如果为None则从pkl文件读取
        """
        print(f"\n🔄 Converting {pkl_path} to SMPL-X format...")
        
        # 读取PKL文件
        with open(pkl_path, 'rb') as f:
            robot_data = pickle.load(f)
        
        root_pos = robot_data['root_pos']  # (N, 3)
        root_rot = robot_data['root_rot']  # (N, 4) xyzw
        dof_pos = robot_data['dof_pos']    # (N, M)
        
        num_frames = len(root_pos)
        print(f"   Frames: {num_frames}")
        
        # 获取fps
        if fps is None:
            fps = robot_data.get('fps', 30.0)
        print(f"   FPS: {fps}")
        
        # 构建完整的qpos序列
        print("📊 Extracting robot joint poses...")
        smplx_joints_list = []
        for i in tqdm(range(num_frames), desc="Processing frames"):
            # 组装qpos: [root_pos(3), root_rot(4 xyzw), dof_pos(M)]
            # 注意：需要转换为wxyz格式给MuJoCo
            qpos = np.concatenate([
                root_pos[i],
                root_rot[i][[3,0,1,2]],  # xyzw -> wxyz
                dof_pos[i]
            ])
            
            # 转换为SMPL-X关节
            smplx_joints = self.robot_frame_to_smplx_joints(qpos)
            smplx_joints_list.append(smplx_joints)
        
        # 转换为SMPL-X参数
        print("🔧 Fitting SMPL-X parameters...")
        smplx_params = self._joints_to_smplx_params(smplx_joints_list, betas)
        
        # 保存为NPZ文件
        print(f"💾 Saving to {output_npz_path}")
        np.savez(
            output_npz_path,
            betas=smplx_params['betas'],
            pose_body=smplx_params['pose_body'],
            root_orient=smplx_params['root_orient'],
            trans=smplx_params['trans'],
            gender=np.array("neutral"),
            mocap_frame_rate=np.array(fps),
        )
        
        print(f"✅ Conversion complete!")
        print(f"   Output: {output_npz_path}")
        print(f"   Frames: {len(smplx_params['pose_body'])}")
        
        return output_npz_path


def main():
    parser = argparse.ArgumentParser(
        description="Reverse Retargeting: Convert robot motion (PKL) to SMPL-X (NPZ)"
    )
    
    parser.add_argument(
        "--pkl_file",
        type=str,
        required=True,
        help="Input robot motion PKL file"
    )
    
    parser.add_argument(
        "--output_npz",
        type=str,
        default=None,
        help="Output SMPL-X NPZ file (default: same as input with .npz extension)"
    )
    
    parser.add_argument(
        "--robot",
        type=str,
        default="unitree_g1",
        choices=["unitree_g1", "unitree_h1", "unitree_h1_2"],
        help="Robot type"
    )
    
    parser.add_argument(
        "--gender",
        type=str,
        default="neutral",
        choices=["male", "female", "neutral"],
        help="SMPL-X gender"
    )
    
    parser.add_argument(
        "--betas",
        type=float,
        nargs=16,
        default=None,
        help="SMPL-X body shape parameters (16 values)"
    )
    
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frame rate (if not specified in PKL)"
    )
    
    args = parser.parse_args()
    
    # 设置路径
    HERE = pathlib.Path(__file__).parent
    ASSETS_PATH = HERE / "assets"
    
    # 机器人XML路径映射
    ROBOT_XML_MAP = {
        "unitree_g1": ASSETS_PATH / "robots" / "unitree_g1" / "scene.xml",
        "unitree_h1": ASSETS_PATH / "robots" / "unitree_h1" / "scene.xml",
        "unitree_h1_2": ASSETS_PATH / "robots" / "unitree_h1_2" / "scene.xml",
    }
    
    # IK配置路径映射
    IK_CONFIG_MAP = {
        "unitree_g1": HERE.parent / "general_motion_retargeting" / "ik_configs" / "smplx_to_g1.json",
        "unitree_h1": HERE.parent / "general_motion_retargeting" / "ik_configs" / "smplx_to_h1.json",
        "unitree_h1_2": HERE.parent / "general_motion_retargeting" / "ik_configs" / "smplx_to_h1.json",
    }
    
    SMPLX_MODEL_PATH = HERE / "assets" / "body_models"
    
    # 检查文件是否存在
    if not pathlib.Path(args.pkl_file).exists():
        print(f"❌ Error: PKL file not found: {args.pkl_file}")
        return
    
    # 设置输出路径
    if args.output_npz is None:
        args.output_npz = str(pathlib.Path(args.pkl_file).with_suffix('.npz'))
    
    # 初始化转换器
    converter = RobotToSMPLX(
        robot_type=args.robot,
        robot_xml_path=ROBOT_XML_MAP[args.robot],
        ik_config_path=IK_CONFIG_MAP[args.robot],
        smplx_model_path=SMPLX_MODEL_PATH,
        gender=args.gender
    )
    
    # 执行转换
    betas = np.array(args.betas) if args.betas else None
    converter.convert_pkl_to_npz(
        pkl_path=args.pkl_file,
        output_npz_path=args.output_npz,
        betas=betas,
        fps=args.fps
    )


if __name__ == "__main__":
    main()





