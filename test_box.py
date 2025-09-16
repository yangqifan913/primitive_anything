+……#!/usr/bin/env python3
"""
等效 box 可视化（Rerun）
- 随机生成 10 个 box (position, rpy, size)
- 枚举 6 种 size 的排列 (l,w,h 的置换)
- 计算等效 box（调整旋转，使其在世界坐标下与原 box 重合）
- 用 rerun 可视化：原始 box 实心，等效 box 线框
- 包含数值校验（角点比较）
"""

import numpy as np
from itertools import permutations
from scipy.spatial.transform import Rotation as R
import rerun as rr

# ------------------------
# 工具函数
# ------------------------
def rotation_matrix_from_rpy(rpy):
    """rpy = [roll, pitch, yaw] (rad), 顺序 'xyz'"""
    return R.from_euler("xyz", rpy).as_matrix()

def quat_xyzw_from_matrix(mat):
    """返回 SciPy 格式的四元数 [x,y,z,w]"""
    return R.from_matrix(mat).as_quat()  # SciPy 保证是 [x,y,z,w]

def equivalent_box_from_columns(position, rpy, size, perm):
    """
    计算等效 box（更稳健的方法）
    参数:
        position: (3,) array-like
        rpy: (3,) array-like, radians, euler xyz (roll,pitch,yaw)
        size: (3,) array-like, [l, w, h]
        perm: tuple of 3 ints, 表示 new_axis_i 对应 old_axis perm[i]
              例如 perm=(1,0,2) 表示 new_x=old_y, new_y=old_x, new_z=old_z
    返回:
        center: (3,)
        half_sizes: (3,)  (注意 Rerun 的 Boxes3D 接受的是 half_sizes)
        quat_xyzw: (4,) 四元数按 [x,y,z,w]
        euler_xyz: (3,) rpy（便于打印 / 调试）
        was_reflection_fix: bool (是否做了 -1 列修正)
    """
    pos = np.array(position, dtype=float)
    size = np.array(size, dtype=float)
    Rm = rotation_matrix_from_rpy(rpy)            # world_from_local 的矩阵，列是局部 x,y,z 在世界坐标里的向量

    # 新旋转矩阵的列直接从原矩阵按 permute 取列
    Rnew = Rm[:, list(perm)].copy()

    # 如果置换导致左手系（det < 0），把第 3 列乘 -1 修正为右手系
    was_reflection_fix = False
    if np.linalg.det(Rnew) < 0:
        Rnew[:, 2] *= -1.0
        was_reflection_fix = True

    new_size = size[list(perm)]
    half_sizes = new_size / 2.0

    quat_xyzw = quat_xyzw_from_matrix(Rnew)
    euler_xyz = R.from_matrix(Rnew).as_euler("xyz")

    return pos, half_sizes.tolist(), quat_xyzw.tolist(), euler_xyz.tolist(), was_reflection_fix

def verify_equivalence(position, rpy, size, perm, pos_new, half_sizes_new, Rnew):
    """
    数值校验：对于原 box 的每个角点，检查能否在新 box 的 8 个角点里找到相同点（绝对误差很小）
    Rnew: rotation matrix used for new box (world_from_local)
    """
    pos = np.array(position, dtype=float)
    size = np.array(size, dtype=float)
    Rm = rotation_matrix_from_rpy(rpy)
    # 原 box 角点
    corners_old = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                local = np.array([sx, sy, sz]) * (size / 2.0)
                corners_old.append(pos + Rm.dot(local))
    corners_old = np.stack(corners_old, axis=0)

    # 新 box 角点
    new_size = np.array(half_sizes_new) * 2.0
    corners_new = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                local = np.array([sx, sy, sz]) * (new_size / 2.0)
                corners_new.append(np.array(pos_new) + Rnew.dot(local))
    corners_new = np.stack(corners_new, axis=0)

    # 对于每个旧角点，检查 corners_new 中是否存在匹配（顺序不一定一样）
    for p in corners_old:
        dists = np.linalg.norm(corners_new - p[None, :], axis=1)
        if np.min(dists) > 1e-6:
            return False
    return True

# ------------------------
# 可视化主逻辑
# ------------------------
def main():
    rr.init("equivalent-box-demo", spawn=True)

    np.random.seed(23)

    n_boxes = 10
    perms = list(permutations([0, 1, 2]))  # 6 种

    # 颜色池（原始一个颜色，6 个等效不同颜色）
    equiv_colors = [
        (0, 200, 200),
        (200, 0, 200),
        (200, 200, 0),
        (0, 150, 0),
        (0, 0, 200),
        (150, 75, 0),
    ]

    for i in range(n_boxes):
        # 随机生成 box
        pos = np.random.uniform(-4.0, 4.0, size=3).tolist()
        # rpy 取范围 [-pi, pi]
        rpy = np.random.uniform(-np.pi, np.pi, size=3).tolist()
        size = np.random.uniform(0.4, 2.0, size=3).tolist()  # l,w,h

        # 原始 box 的 quaternion
        Rm = rotation_matrix_from_rpy(rpy)
        quat_orig = quat_xyzw_from_matrix(Rm).tolist()
        half_orig = (np.array(size) / 2.0).tolist()

        # log 原始 box（实心，红色）
        rr.log(
            f"box_{i}/original",
            rr.Boxes3D(
                centers=[pos],
                half_sizes=[half_orig],
                quaternions=[rr.Quaternion(xyzw=quat_orig)],
                colors=[(255, 50, 50)],
                # fill_mode=rr.FillMode.Solid,
                labels=[f"box_{i} orig"],
            ),
        )

        # 所有等效 box（线框）
        print(perms)
        for j, perm in enumerate(perms):
            pos_new, half_new, quat_new, euler_new, fixed = equivalent_box_from_columns(pos, rpy, size, perm)
            # 用 Rnew 做数值校验（构造 matrix）
            print('**********',half_new)
            Rnew_mat = R.from_quat(quat_new).as_matrix()

            ok = verify_equivalence(pos, rpy, size, perm, pos_new, half_new, Rnew_mat)
            if not ok:
                # 这不应该发生 — 打印调试信息
                print(f"[WARN] verification failed for box {i} perm {perm}")

            color = equiv_colors[j % len(equiv_colors)]
            rr.log(
                f"box_{i}/equiv_{j}",
                rr.Boxes3D(
                    centers=[pos_new],
                    half_sizes=[half_new],
                    quaternions=[rr.Quaternion(xyzw=quat_new)],
                    colors=[color],
                    # fill_mode=rr.FillMode.MajorWireframe,
                    labels=[f"perm={perm}{' (fix)' if fixed else ''}"],
                ),
            )

    print("已 log 完成：10 个 box 及其等效 box，打开 Rerun Viewer（spawn=True 已自动打开）查看。")

if __name__ == "__main__":
    main()
