from scipy.spatial.transform import Rotation as R

#检查从 smplx_to_g1.json中reverse 计算得到的 g1_to_smplx_json的rot_offset是否正确

#python scripts/test_qua_compuation.py

#mark: smplx_to_g1.json中的quat是wxyz的顺序
 
# w, x, y, z = 0.4267755048530407,-0.5637931078484661,-0.5637931078484661,-0.4267755048530407
# w, x, y, z =  0.5, -0.5,-0.5, -0.5
w, x, y, z =  -0.5, 0.5, 0.5, 0.5
# w, x, y, z =  0.70710678, 0.0, -0.70710678, 0.0
# w, x, y, z =  0, 0.70710678, 0, 0.70710678
# w, x, y, z =  1, 0, 0, 0
# w, x, y, z =  0, 0, 0, -1
tx, ty, tz = 0.0, 0.0, 0.0
# tx, ty, tz = 0.0, 0.02, 0.0
# tx, ty, tz = 0.0, -0.02, 0.0

rot_offset = R.from_quat([w, x, y, z], scalar_first=True)  # wxyz
pos_offset = [tx, ty, tz]
pos_offset_rev = - rot_offset.apply(pos_offset) #core calculation
print(pos_offset_rev)
# print(-R_off.apply([tx, ty, tzx]))

#ik1 ✅
#[-0. -0. -0.]
#[-0. -0. -0.]
#[-0. -0. -0.]
#[-0.02 -0. -0.]
#[-0. -0. -0.]
#[-0. -0. -0.]
#[0.02 -0. -0.]
#[-0. -0. -0.]
#[-0. -0. -0.]
#[-0. -0. -0.]   left_elbow
#[-0. -0. -0.]
#[-0. -0. -0.]   right_shoulder_yaw_link
#[-0. -0. -0.]
#[-0. -0. -0.]


# ik2 (只有两个需要另外计算)
#left_toe_link  [-0. -0. -0.] ✅
#right_toe_link [-0. -0. -0.] ✅




