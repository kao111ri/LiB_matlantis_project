"""
Solvation Utilities for Filling Structures with Solvent Molecules

このモジュールは、既存の構造（LiPF6、PVDF、基板など）の空隙を
溶媒分子（水など）で充填するための関数を提供します。

主な機能:
- 密度指定による溶媒分子数の自動計算
- ランダム配置と重なりチェック
- 様々な分子種（H2O、その他）への対応
"""

import numpy as np
from typing import Optional, Tuple
from ase import Atoms, units
from ase.build import molecule
from ase.geometry import get_distances


def fill_box_with_molecules(
    host_atoms: Atoms,
    solvent_type: str = 'H2O',
    density_g_cm3: float = 1.0,
    min_distance: float = 2.0,
    max_attempts_per_molecule: int = 1000,
    target_fill_fraction: float = 1.0,
    random_seed: Optional[int] = None,
    verbose: bool = True
) -> Atoms:
    """
    既存の構造の空隙を溶媒分子で満たす汎用関数

    Parameters
    ----------
    host_atoms : Atoms
        既存のAtomsオブジェクト（基板、LiPF6、PVDFなど）
    solvent_type : str
        溶媒分子の種類 ('H2O', 'CH3OH', 'C2H5OH'など、ase.build.moleculeで認識される名前)
    density_g_cm3 : float
        ターゲットとする溶媒の密度 (g/cm^3)
        - 水: 1.0 g/cm^3 (25°C)
        - より緩い初期配置には 0.8-0.9 g/cm^3 を推奨
    min_distance : float
        既存原子と溶媒分子原子の最小許容距離 (Å)
    max_attempts_per_molecule : int
        各分子の配置試行回数の上限
    target_fill_fraction : float
        目標分子数の達成率（0.0-1.0）
        - 1.0: 全分子を配置しようとする（時間がかかる）
        - 0.8-0.9: 80-90%配置できればOK（推奨）
    random_seed : int, optional
        乱数シード（再現性のため）
    verbose : bool
        進捗メッセージを表示するか

    Returns
    -------
    Atoms
        溶媒分子を追加した新しいAtomsオブジェクト

    Examples
    --------
    >>> from ase.io import read
    >>> structure = read('LiPF6.cif')
    >>> filled = fill_box_with_molecules(structure, 'H2O', density_g_cm3=0.9)
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    # 1. 溶媒分子のテンプレート作成
    try:
        solvent_template = molecule(solvent_type)
    except Exception as e:
        raise ValueError(f"溶媒分子 '{solvent_type}' を作成できません: {e}")

    # 溶媒分子の質量を計算 (g/mol)
    solvent_mass_amu = solvent_template.get_masses().sum()
    solvent_molar_mass = solvent_mass_amu  # AMU = g/mol

    if verbose:
        print(f"\n=== 溶媒充填開始 ===")
        print(f"溶媒分子: {solvent_type}")
        print(f"分子量: {solvent_molar_mass:.3f} g/mol")
        print(f"分子あたり原子数: {len(solvent_template)}")

    # 2. セル体積の計算
    vol_A3 = host_atoms.get_volume()
    vol_cm3 = vol_A3 * 1.0e-24  # Å^3 -> cm^3

    if verbose:
        print(f"\nセル体積: {vol_A3:.2f} Å³ ({vol_cm3:.3e} cm³)")

    # 3. 必要な溶媒分子数の計算
    # 密度 (g/cm³) * 体積 (cm³) / モル質量 (g/mol) * アボガドロ数 -> 分子数
    total_mass_g = density_g_cm3 * vol_cm3
    n_moles = total_mass_g / solvent_molar_mass
    n_molecules_target = int(n_moles * units.mol)

    if verbose:
        print(f"\n密度: {density_g_cm3} g/cm³")
        print(f"目標分子数: {n_molecules_target}")
        print(f"目標原子数: {n_molecules_target * len(solvent_template)}")

    # 4. 既存原子の座標を取得
    host_positions = host_atoms.get_positions()
    cell = host_atoms.get_cell()
    pbc = host_atoms.get_pbc()

    if verbose:
        print(f"\n既存構造:")
        print(f"  組成: {host_atoms.get_chemical_formula()}")
        print(f"  原子数: {len(host_atoms)}")
        print(f"  PBC: {pbc}")

    # 5. 溶媒分子の配置ループ
    solvent_atoms = Atoms(cell=cell, pbc=pbc)
    n_placed = 0
    n_target_actual = int(n_molecules_target * target_fill_fraction)

    if verbose:
        print(f"\n配置開始 (目標: {n_target_actual} 分子)...")

    for i_mol in range(n_molecules_target):
        placed = False

        for i_attempt in range(max_attempts_per_molecule):
            # ランダムな位置と回転
            mol = solvent_template.copy()

            # セル内のランダムな位置へ移動（スケール座標）
            scaled_pos = np.random.random(3)
            real_pos = np.dot(scaled_pos, cell)

            # 分子の重心を移動
            mol_com = mol.positions.mean(axis=0)
            mol.translate(real_pos - mol_com)

            # ランダム回転
            mol.rotate(np.random.rand() * 360, 'z', center=real_pos)
            mol.rotate(np.random.rand() * 360, 'x', center=real_pos)
            mol.rotate(np.random.rand() * 360, 'y', center=real_pos)

            # 周期境界条件でラップ
            mol.wrap()

            # 6. 重なりチェック
            mol_positions = mol.get_positions()
            overlap = False

            # 既存構造との距離チェック
            if len(host_atoms) > 0:
                for mol_pos in mol_positions:
                    dists, _ = get_distances(
                        mol_pos[np.newaxis, :],
                        host_positions,
                        cell=cell,
                        pbc=pbc
                    )
                    if np.min(dists) < min_distance:
                        overlap = True
                        break

            # 既に配置した溶媒分子との距離チェック
            if not overlap and len(solvent_atoms) > 0:
                solvent_positions = solvent_atoms.get_positions()
                for mol_pos in mol_positions:
                    dists, _ = get_distances(
                        mol_pos[np.newaxis, :],
                        solvent_positions,
                        cell=cell,
                        pbc=pbc
                    )
                    if np.min(dists) < min_distance:
                        overlap = True
                        break

            # 配置成功
            if not overlap:
                solvent_atoms.extend(mol)
                n_placed += 1
                placed = True

                if verbose and n_placed % max(1, n_target_actual // 10) == 0:
                    progress = (n_placed / n_target_actual) * 100
                    print(f"  配置完了: {n_placed}/{n_target_actual} ({progress:.1f}%)")

                break

        # 目標達成率に到達したら終了
        if n_placed >= n_target_actual:
            break

        # 配置失敗が続いたら警告
        if not placed and verbose and i_mol % 100 == 0:
            print(f"  警告: 分子 {i_mol+1} を {max_attempts_per_molecule} 回試行しても配置できませんでした")

    # 7. 結果の統計
    if verbose:
        print(f"\n=== 充填完了 ===")
        print(f"配置成功: {n_placed}/{n_molecules_target} 分子 ({n_placed/n_molecules_target*100:.1f}%)")
        print(f"追加原子数: {len(solvent_atoms)}")

        # 実際の密度を計算
        actual_mass_g = (n_placed * solvent_molar_mass) / units.mol
        actual_density = actual_mass_g / vol_cm3
        print(f"実際の密度: {actual_density:.3f} g/cm³ (目標: {density_g_cm3:.3f} g/cm³)")

    # 8. 結合して返す
    final_atoms = host_atoms + solvent_atoms
    final_atoms.set_cell(cell)
    final_atoms.set_pbc(pbc)

    # 既存の制約を保持
    if host_atoms.constraints:
        final_atoms.set_constraint(host_atoms.constraints)

    return final_atoms


def fill_lipf6_with_water(
    lipf6_atoms: Atoms,
    water_density: float = 0.9,
    min_distance: float = 2.2,
    **kwargs
) -> Atoms:
    """
    LiPF6構造を水で充填する専用関数

    Parameters
    ----------
    lipf6_atoms : Atoms
        LiPF6構造のAtomsオブジェクト
    water_density : float
        水の密度 (g/cm^3)、デフォルトは0.9（緩めの初期配置）
    min_distance : float
        最小距離 (Å)、LiPF6のFと水のHの距離を考慮して2.2Å推奨
    **kwargs
        fill_box_with_moleculesへの追加引数

    Returns
    -------
    Atoms
        水を追加したLiPF6系

    Examples
    --------
    >>> lipf6 = read('/home/jovyan/Kaori/MD/input/LiPF6.cif')
    >>> solvated = fill_lipf6_with_water(lipf6)
    """
    print("=" * 60)
    print("  LiPF6構造への水分子充填")
    print("=" * 60)

    return fill_box_with_molecules(
        lipf6_atoms,
        solvent_type='H2O',
        density_g_cm3=water_density,
        min_distance=min_distance,
        **kwargs
    )


def fill_pvdf_with_water(
    pvdf_atoms: Atoms,
    water_density: float = 0.9,
    min_distance: float = 2.2,
    **kwargs
) -> Atoms:
    """
    PVDF構造を水で充填する専用関数

    Parameters
    ----------
    pvdf_atoms : Atoms
        PVDF構造のAtomsオブジェクト
    water_density : float
        水の密度 (g/cm^3)、デフォルトは0.9（緩めの初期配置）
    min_distance : float
        最小距離 (Å)、PVDFのFと水のHの距離を考慮して2.2Å推奨
    **kwargs
        fill_box_with_moleculesへの追加引数

    Returns
    -------
    Atoms
        水を追加したPVDF系

    Examples
    --------
    >>> pvdf = read('PVDF_only_shrunk.cif')
    >>> solvated = fill_pvdf_with_water(pvdf)
    """
    print("=" * 60)
    print("  PVDF構造への水分子充填")
    print("=" * 60)

    return fill_box_with_molecules(
        pvdf_atoms,
        solvent_type='H2O',
        density_g_cm3=water_density,
        min_distance=min_distance,
        **kwargs
    )


def estimate_required_molecules(
    atoms: Atoms,
    solvent_type: str = 'H2O',
    density_g_cm3: float = 1.0
) -> dict:
    """
    必要な溶媒分子数を事前に見積もる補助関数

    Parameters
    ----------
    atoms : Atoms
        対象構造
    solvent_type : str
        溶媒分子の種類
    density_g_cm3 : float
        目標密度

    Returns
    -------
    dict
        見積もり結果の辞書
    """
    solvent_template = molecule(solvent_type)
    solvent_molar_mass = solvent_template.get_masses().sum()

    vol_A3 = atoms.get_volume()
    vol_cm3 = vol_A3 * 1.0e-24

    total_mass_g = density_g_cm3 * vol_cm3
    n_moles = total_mass_g / solvent_molar_mass
    n_molecules = int(n_moles * units.mol)
    n_atoms = n_molecules * len(solvent_template)

    return {
        'volume_A3': vol_A3,
        'volume_cm3': vol_cm3,
        'target_density_g_cm3': density_g_cm3,
        'solvent_type': solvent_type,
        'solvent_molar_mass': solvent_molar_mass,
        'n_molecules_required': n_molecules,
        'n_atoms_required': n_atoms,
        'existing_atoms': len(atoms),
        'total_atoms': len(atoms) + n_atoms,
    }


if __name__ == "__main__":
    # テスト用コード
    from ase.build import bulk
    from ase.io import write

    print("\n" + "=" * 60)
    print("  Solvation Utils テスト")
    print("=" * 60)

    # テスト用のAl構造を作成
    test_structure = bulk('Al', 'fcc', a=4.05).repeat((3, 3, 3))
    test_structure.center(vacuum=5.0, axis=2)

    print("\nテスト構造:")
    print(f"  組成: {test_structure.get_chemical_formula()}")
    print(f"  セル: {test_structure.cell.cellpar()}")

    # 見積もり
    print("\n--- 必要分子数の見積もり ---")
    estimate = estimate_required_molecules(test_structure, 'H2O', 0.9)
    for key, value in estimate.items():
        print(f"  {key}: {value}")

    # 充填実行
    filled = fill_box_with_molecules(
        test_structure,
        solvent_type='H2O',
        density_g_cm3=0.9,
        min_distance=2.2,
        target_fill_fraction=0.8,
        random_seed=42
    )

    # 保存
    output_file = 'test_filled_structure.xyz'
    write(output_file, filled)
    print(f"\n結果を保存: {output_file}")
    print(f"最終組成: {filled.get_chemical_formula()}")
