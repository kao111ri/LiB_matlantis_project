"""
構造操作関連のユーティリティ関数

このモジュールには以下の機能が含まれます：
- 界面構築
- 密度計算
- セル設定
"""

import numpy as np
from ase import units
from typing import Tuple


def build_interface(
    slab1,
    slab2,
    target_xy: Tuple[float, float] = (12.0, 12.0),
    separation: float = 2.0
):
    """
    2つのスラブから界面を構築する

    Args:
        slab1: 下層のASE Atomsオブジェクト
        slab2: 上層のASE Atomsオブジェクト
        target_xy (tuple): xy平面のターゲットサイズ (Å)
        separation (float): 2つのスラブ間の距離 (Å)

    Returns:
        界面構造のASE Atomsオブジェクト
    """
    print(f"    1. 界面を構築中...")

    position1 = slab1.get_positions()
    position2 = slab2.get_positions()

    # xy平面で範囲外の原子を削除
    indices_to_delete1 = np.where(
        (position1[:, 0] >= target_xy[0]) |
        (position1[:, 0] < 0) |
        (position1[:, 1] >= target_xy[1]) |
        (position1[:, 1] < 0)
    )[0]
    indices_to_delete2 = np.where(
        (position2[:, 0] >= target_xy[0]) |
        (position2[:, 1] >= target_xy[1]) |
        (position2[:, 0] < 0) |
        (position2[:, 1] < 0)
    )[0]

    cut_slab1, cut_slab2 = slab1.copy(), slab2.copy()
    if len(indices_to_delete1) > 0:
        del cut_slab1[indices_to_delete1]
    if len(indices_to_delete2) > 0:
        del cut_slab2[indices_to_delete2]

    # z方向の位置を調整
    z1_max = cut_slab1.positions[:, 2].max()
    z2_min = cut_slab2.positions[:, 2].min()
    cut_slab2.positions[:, 2] += z1_max - z2_min + separation

    # 2つのスラブを結合
    interface = cut_slab1 + cut_slab2

    return interface


def calculate_density(atoms, num_molecules: int, molar_mass: float) -> float:
    """
    構造の密度を計算する

    Args:
        atoms: ASE Atomsオブジェクト
        num_molecules (int): 分子数
        molar_mass (float): 分子のモル質量 (g/mol)

    Returns:
        密度 (g/cm³)
    """
    # 総質量 (g)
    total_mass = num_molecules * molar_mass / units.mol

    # 体積 (cm³)
    volume_cm3 = atoms.get_volume() * 1e-24

    # 密度 (g/cm³)
    density = total_mass / volume_cm3

    return density


def calculate_cell_from_density(
    num_molecules: int,
    molar_mass: float,
    target_density: float
) -> float:
    """
    目標密度から立方体セルの一辺の長さを計算する

    Args:
        num_molecules (int): 分子数
        molar_mass (float): 分子のモル質量 (g/mol)
        target_density (float): 目標密度 (g/cm³)

    Returns:
        セルの一辺の長さ (Å)
    """
    # 総質量 (g/mol)
    total_mass = molar_mass * num_molecules

    # 体積 (Å³) = (総質量[g/mol] / アボガドロ数) / (密度[g/cm³]) * (1e24 [Å³/cm³])
    volume_A3 = (total_mass / units.mol) / (target_density / 1e24)

    # 立方体セルの一辺の長さ
    cell_side_length = volume_A3 ** (1/3)

    return cell_side_length


def set_cell_with_vacuum(atoms, vacuum: float = 0.2):
    """
    分子の範囲に基づいてセルを設定し、真空層を追加する

    Args:
        atoms: ASE Atomsオブジェクト
        vacuum (float): 真空層の厚さ (Å)

    Returns:
        セルが設定されたASE Atomsオブジェクト
    """
    positions = atoms.get_positions()
    min_coords = positions.min(axis=0)
    max_coords = positions.max(axis=0)

    # セルの大きさを計算（分子の最大幅 + 両側の真空層）
    cell_lengths = (max_coords - min_coords) + 2 * vacuum
    new_cell = np.diag(cell_lengths)

    atoms.set_cell(new_cell)
    atoms.pbc = True

    return atoms


def create_water_unit_cell(target_density: float = 1.0) -> 'ase.Atoms':
    """
    指定密度の水分子単位セルを作成する

    Args:
        target_density (float): 目標密度 (g/cm³)

    Returns:
        水分子単位セルのASE Atomsオブジェクト
    """
    from ase.build import molecule

    # 定数
    molar_mass_h2o = 2 * 1.008 + 15.999  # 水のモル質量 (g/mol)
    angstrom3_to_cm3 = (1e-8) ** 3

    # 単位セルの体積計算
    mass_per_molecule_g = molar_mass_h2o / 6.022e23
    target_density_g_angstrom3 = target_density * angstrom3_to_cm3
    volume_per_molecule_angstrom3 = mass_per_molecule_g / target_density_g_angstrom3

    # 立方体セルの辺の長さ
    side_length = volume_per_molecule_angstrom3 ** (1/3.0)
    cell_vec = [side_length, side_length, side_length]

    # 単位セルの構築
    unit_cell = molecule("H2O")
    unit_cell.set_cell(cell_vec)
    unit_cell.center()
    unit_cell.pbc = True

    print(f"水分子単位セル作成: 密度 {target_density} g/cm³, セル辺 {side_length:.2f} Å")

    return unit_cell
