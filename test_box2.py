#!/usr/bin/env python3
"""
修正后的脚本：生成 10 个随机 box，每个输出 24 个方向保持的等效 box（几何上重合）
并使用 rerun 0.21.0 可视化。脚本会对每个等效 box 做数值验证（角点集合一致性）。
"""

import numpy as np
from itertools import permutations, product
from scipy.spatial.transform import Rotation as R
import rerun as rr

np.set_printoptions(precision=6, suppress=True)


def signed_perm_matrices(det_keep=1):
    """生成所有 signed permutation 矩阵，保留 det == det_keep（det_keep=+1 -> 24 个）"""
    mats = []
    for perm in permutations([0, 1, 2]):
        for signs in product([1, -1], repeat=3):
            S = np.zeros((3, 3), dtype=int)
            for i, p in enumerate(perm):
                S[i, p] = signs[i]
            d = int(round(np.linalg.det(S)))
            if d == det_keep:
                mats.append(S)
    return mats


def corners_from(center, Rmat, half_sizes):
    """返回 box 在世界坐标中的 8 个角点 (shape (8,3))"""
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                local = np.array([sx * half_sizes[0], sy * half_sizes[1], sz * half_sizes[2]])
                corners.append(center + Rmat.dot(local))
    return np.stack(corners, axis=0)


def almost_same_corner_sets(A, B, tol=1e-6):
    """
    检查角点集合 A 与 B 是否匹配（顺序可不同）。
    对 A 中每个点找 B 的最小距离，最大最小距离需 <= tol。
    """
    assert A.shape == B.shape and A.shape[1] == 3
    maxmin = 0.0
    for a in A:
        dists = np.linalg.norm(B - a[None, :], axis=1)
        min_d = float(np.min(dists))
        if min_d > maxmin:
            maxmin = min_d
        if min_d > tol:
            return False, maxmin
    return True, maxmin




def generate_equivalents(center, rpy_deg, size):
    """
    对给定 box（center, rpy_deg, size）生成 24 个 orientation-preserving 等效 (R', half')
    返回列表：[(Rmat', half_sizes'), ...]
    """
    half = np.array(size, dtype=float) / 2.0
    Rmat = R.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    Ss = signed_perm_matrices(det_keep=1)  # 24 个 S

    results = []
    for S in Ss:
        # R' = R * S^T
        Rprime = Rmat @ S.T
        half_prime = np.abs(S @ half)  # positive half sizes
        results.append((Rprime, half_prime, S))
    return results


def log_box_with_transform(path, center, half_sizes, Rmat, color=(255, 0, 0), wireframe=False):
    """在 rerun 中按 path log Boxes3D（直接在Boxes3D中包含旋转）"""
    # 将旋转矩阵转换为四元数
    quat = R.from_matrix(Rmat).as_quat()  # [x, y, z, w]
    
    # Boxes3D expects half_sizes shape (N,3) and centers shape (N,3)
    rr.log(
        path,
        rr.Boxes3D(
            centers=np.array([center]),
            half_sizes=np.array([half_sizes]),
            quaternions=[rr.Quaternion(xyzw=quat)],  # 直接在Boxes3D中包含旋转
            colors=[color],
        ),
    )


def main():
    rr.init("equivalent-boxes-fixed", spawn=True)
    np.random.seed(123)

    # 彩色池（原始 green，等效使用多个颜色）
    equiv_colors = [
        (200, 50, 50),
        (200, 120, 50),
        (200, 200, 50),
        (120, 200, 50),
        (50, 200, 120),
        (50, 120, 200),
        (120, 50, 200),
        (200, 50, 120),
    ]

    # 生成 10 个随机 box 并可视化
    for b in range(10):
        center = np.random.uniform(-2.0, 2.0, size=3)
        size = np.random.uniform(0.5, 1.5, size=3)  # l,w,h
        rpy = np.random.uniform(0.0, 360.0, size=3)

        print(f"\n原始 box {b}:")
        print(" pos =", np.array2string(center, precision=6))
        print(" size =", np.array2string(size, precision=6))
        print(" rpy  =", np.array2string(rpy, precision=6))

        # 原始旋转矩阵与角点
        Rmat = R.from_euler("xyz", rpy, degrees=True).as_matrix()
        half = size / 2.0
        corners_orig = corners_from(center, Rmat, half)

        # log 原始 box（绿色实心）
        log_box_with_transform(f"box{b}/original", center, half, Rmat, color=(0, 200, 0), wireframe=False)

        # 产生 24 个等效
        equivs = generate_equivalents(center, rpy, size)
        ok_count = 0
        for i, (Rprime, half_prime, S) in enumerate(equivs):
            corners_new = corners_from(center, Rprime, half_prime)
            ok, worst = almost_same_corner_sets(corners_orig, corners_new, tol=1e-6)
            if ok:
                ok_count += 1
            else:
                print(f"[WARN] box{b} equiv {i} failed verification, worst min-dist = {worst:.3e}, S=\n{S}")

            color = equiv_colors[i % len(equiv_colors)]
            log_box_with_transform(f"box{b}/equiv/{i}", center, half_prime, Rprime, color=color, wireframe=True)

        print(f" Verification: {ok_count}/{len(equivs)} equivalents matched (should be 24).")

    print("\n全部日志已写入 rerun。打开 Rerun Viewer 检查每个 box 的 original/equiv 层级（wireframe 为等效 box）。")


if __name__ == "__main__":
    main()




