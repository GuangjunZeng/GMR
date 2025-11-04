#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os


# python scripts/view_npz.py  assets/body_models/smplx/SMPLX_MALE.npz
# python scripts/view_npz.py  ../server3_data/locomotion/reference/000001.npz
# python scripts/view_npz.py  ../server3_data/locomotion/reference/000002.npz
# python scripts/view_npz.py  ../server3_data/locomotion/reference/010220.npz


# 文件包含的键: ['gender', 'betas', 'pose_body', 'pose_hand', 'smpl_trans', 'smpl_quat_xyzw', 'pelvis_trans', 'pelvis_quat_xyzw', 'joints_local', 'fps']

def view_npz_data(npz_path, show_preview=True, save_csv=False):
    """查看NPZ文件内容"""
    print(f"🔍 查看NPZ文件: {npz_path}")
    print("="*60)
    
    # 加载NPZ文件
    data = np.load(npz_path, allow_pickle=True)
    
    print(f"📁 文件包含的键: {list(data.keys())}")
    print()
    
    for key in data.keys():
        value = data[key]
        print(f"🔑 {key}:")
        
        if isinstance(value, np.ndarray):
            print(f"   形状: {value.shape}")
            print(f"   数据类型: {value.dtype}")

            if key == 'gender':
                if value.size == 1:
                    print(f"   值: {value.item()}")
                else:
                    print(f"   值列表: {value.tolist()}")
                print()
                continue
            
            # 处理对象类型（如字典、列表等）
            if value.dtype == object:
                print(f"   类型: 对象 (object)")
                if value.size == 1:
                    obj = value.item()
                    if isinstance(obj, dict):
                        print(f"   字典内容: {obj}")
                    else:
                        print(f"   值: {obj}")
                elif value.size > 0 and show_preview:
                    print(f"   前几个值: {value.flatten()[:5]}")
            else:
                # 数值类型才计算范围
                try:
                    print(f"   数值范围: [{np.min(value):.6f}, {np.max(value):.6f}]")
                except Exception:
                    print(f"   无法计算数值范围")

                if key == 'betas':
                    if value.size > 0:
                        pass
                        # print(f"   全部数值: {np.array2string(value, precision=6, separator=', ')}")
                if show_preview and value.size > 0:
                    if value.ndim == 1:
                        print(f" 全部数值 : {value[:value.size]}")
    
            
            # 如果是关节名称
            if key == 'joint_names':
                print(f"   关节名称: {list(value)}")
        else:
            print(f"   值: {value}")
        print()
    
    # 保存为CSV（如果请求）
    if save_csv and 'full_data' in data:
        csv_path = npz_path.replace('.npz', '_extracted.csv')
        np.savetxt(csv_path, data['full_data'], fmt='%.6f', delimiter=',')
        print(f"💾 已保存为CSV: {csv_path}")
    
    data.close()

def main():
    ap = argparse.ArgumentParser("查看NPZ文件内容")
    ap.add_argument("npz_path", help="NPZ文件路径")
    ap.add_argument("--no-preview", action="store_true", help="不显示数据预览")
    ap.add_argument("--save-csv", action="store_true", help="保存为CSV文件")
    args = ap.parse_args()
    
    if not os.path.exists(args.npz_path):
        print(f"❌ 文件不存在: {args.npz_path}")
        return
    
    view_npz_data(args.npz_path, 
                  show_preview=not args.no_preview, 
                  save_csv=args.save_csv)

if __name__ == "__main__":
    main()
