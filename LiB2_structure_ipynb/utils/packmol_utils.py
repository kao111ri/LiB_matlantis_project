"""
Packmol Utilities for ASE

このモジュールは、外部ツール「Packmol」を使用して、
既存の構造（ホスト）の空隙を溶媒分子で充填するためのラッパー関数を提供します。
ASE単独での挿入よりも高速かつ確実に、原子の重なりがない構造を作成できます。

必須要件:
- システムに 'packmol' コマンドがインストールされていること
  (または実行バイナリへのパスを指定すること)

主な機能:
- 密度指定による溶媒分子数の自動計算
- Packmol入力ファイルの自動生成と実行
- 斜交セル（Triclinic）の直方体化とクリッピング機能
- 一時ファイルの自動クリーンアップ
"""

import os
import subprocess
import tempfile
import shutil
import numpy as np
from typing import Optional, Union, List
from ase import Atoms, units
from ase.build import molecule
from ase.io import read, write

# Packmolの実行コマンド（環境に合わせて変更可能）
DEFAULT_PACKMOL_PATH = 'packmol'

def check_packmol_command(cmd: str = DEFAULT_PACKMOL_PATH) -> bool:
    """Packmolコマンドが利用可能かチェックする"""
    return shutil.which(cmd) is not None

def estimate_required_molecules(
    vol_A3: float,
    solvent_molar_mass: float,
    density_g_cm3: float
) -> int:
    """密度と体積から必要な分子数を計算する"""
    vol_cm3 = vol_A3 * 1.0e-24
    total_mass_g = density_g_cm3 * vol_cm3
    n_moles = total_mass_g / solvent_molar_mass
    return int(n_moles * units.mol)

def force_orthorhombic_by_cropping(atoms: Atoms, verbose: bool = True) -> Atoms:
    """
    斜交セル(Triclinic)から、原子密度を維持したまま内接する直方体領域を切り出す。
    
    アルゴリズム:
    1. 斜交セルの各面間隔（有効な厚み）を計算し、切り出す直方体のサイズ(width_a, width_b, width_c)とする。
    2. 元の構造を3x3x3のスーパーセルに拡張し、原子の欠損がない状態を作る。
    3. スーパーセルの中心から、計算したサイズ分の直方体領域を切り出す。
    これにより、斜めの境界による原子の欠けを防ぎ、元の密度を保った直方体ブロックを得る。
    """
    if atoms.cell.orthorhombic:
        return atoms

    # 1. 面間隔（直方体として切り出せる最大の有効幅）の計算
    cell = atoms.get_cell()
    vol = atoms.get_volume()
    
    # 各ベクトルと対向する面の面積を計算 (外積のノルム)
    # Area_bc (aベクトルに対する面) = |b x c|
    area_a = np.linalg.norm(np.cross(cell[1], cell[2])) 
    area_b = np.linalg.norm(np.cross(cell[0], cell[2])) 
    area_c = np.linalg.norm(np.cross(cell[0], cell[1])) 
    
    # 面間隔 (Width) = Volume / Area
    # これが「斜交セルを直方体に圧縮・切断したときの有効サイズ」に相当
    w_a = vol / area_a
    w_b = vol / area_b
    w_c = vol / area_c
    
    target_size = np.array([w_a, w_b, w_c])

    if verbose:
        print(f"警告: 斜交セル(Non-orthorhombic)を検知しました。")
        print(f"-> スーパーセルから内側の直方体領域を切り出します。")
        print(f"   元のセルパラメータ: {atoms.cell.cellpar().round(3)}")
        print(f"   切り出しサイズ (面間隔): {target_size.round(3)}")

    # 2. スーパーセルの作成
    # 斜めの構造をカバーするため、周囲に余裕を持たせる(3x3x3)
    supercell = atoms.repeat((3, 3, 3))
    
    # 3. 中心からの切り出し
    # スーパーセル全体の幾何学的中心
    supercell_center = np.sum(supercell.get_cell(), axis=0) / 2.0
    
    # 切り出し範囲 (min, max)
    limit_min = supercell_center - target_size / 2.0
    limit_max = supercell_center + target_size / 2.0
    
    # 範囲内の原子を抽出
    pos = supercell.get_positions()
    # 各次元ですべて範囲内にある原子のインデックス
    mask = np.all((pos >= limit_min) & (pos < limit_max), axis=1)
    
    new_atoms = supercell[mask].copy()
    
    # 4. 新しいセルの設定
    # 切り出したボックスの左下(limit_min)を原点(0,0,0)に持ってくる
    new_atoms.translate(-limit_min)
    new_atoms.set_cell(np.diag(target_size))
    new_atoms.set_pbc(True)

    if verbose:
        print(f"   処理完了: 元原子数 {len(atoms)} -> スーパーセル切り出し後 {len(new_atoms)}")
        print(f"   新しいセル: {new_atoms.cell.cellpar().round(3)}")

    return new_atoms

def fill_box_with_packmol(
    host_atoms: Atoms,
    solvent_type: str = 'H2O',
    density_g_cm3: float = 1.0,
    tolerance: float = 2.0,
    packmol_path: str = DEFAULT_PACKMOL_PATH,
    keep_temp_files: bool = False,
    verbose: bool = True,
    seed: int = -1
) -> Atoms:
    """
    Packmolを使用して既存構造の隙間に溶媒分子を充填する関数

    Parameters
    ----------
    host_atoms : Atoms
        既存のホスト構造（位置は固定されます）。
        斜交セルの場合は自動的に直方体にクロップされます。
    solvent_type : str
        溶媒分子の種類 ('H2O', 'CH3OH'など、ase.build.moleculeで認識される名前)
    density_g_cm3 : float
        ターゲットとする溶媒の密度 (g/cm^3)
    tolerance : float
        原子間の最小許容距離 (Å)。Packmolのtoleranceパラメータ。2.0程度が推奨。
    packmol_path : str
        packmolの実行コマンドまたはパス
    keep_temp_files : bool
        デバッグ用に一時ファイル(.inp, .xyz)を残すかどうか
    verbose : bool
        詳細な出力を表示するか
    seed : int
        乱数シード (-1の場合はランダム)

    Returns
    -------
    Atoms
        溶媒分子が充填された新しい構造
    """

    # 0. Packmolの存在確認
    if not shutil.which(packmol_path):
        raise FileNotFoundError(
            f"Packmolコマンド '{packmol_path}' が見つかりません。"
            "インストールされているか、パスが正しいか確認してください。"
        )

    if verbose:
        print(f"\n=== Packmolによる充填開始 ===")
        print(f"Packmol path: {packmol_path}")

    # 1. ホスト構造の前処理（斜交セルの場合は直方体に整形）
    # 元のオブジェクトを変更しないようにコピーを使用
    working_host = host_atoms.copy()
    if not working_host.cell.orthorhombic:
        working_host = force_orthorhombic_by_cropping(working_host, verbose=verbose)
    
    # 作業用の一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = "." if keep_temp_files else tmpdir
        
        # ファイル名の定義
        host_xyz = os.path.join(work_dir, "host_temp.xyz")
        solvent_xyz = os.path.join(work_dir, "solvent_temp.xyz")
        output_xyz = os.path.join(work_dir, "packed_output.xyz")
        inp_file = os.path.join(work_dir, "pack.inp")

        # 2. 分子データの準備
        try:
            solvent_atoms = molecule(solvent_type)
        except Exception as e:
            raise ValueError(f"溶媒分子 '{solvent_type}' を作成できません: {e}")

        solvent_mass = solvent_atoms.get_masses().sum()

        # 3. 必要な分子数の計算
        # 整形後のセル情報を使用
        cell_lengths = working_host.cell.cellpar()[:3] # a, b, c
        vol_A3 = working_host.get_volume()
        
        n_molecules = estimate_required_molecules(vol_A3, solvent_mass, density_g_cm3)

        if verbose:
            print(f"溶媒: {solvent_type} (Mass: {solvent_mass:.2f})")
            print(f"整形後セル体積: {vol_A3:.2f} A^3")
            print(f"目標密度: {density_g_cm3:.2f} g/cm^3")
            print(f"計算された分子数: {n_molecules}")

        # 4. ファイルの書き出し
        write(host_xyz, working_host)
        write(solvent_xyz, solvent_atoms)

        # 5. Packmol入力ファイルの作成
        # 直方体領域(0,0,0) -> (a,b,c)を指定
        box_str = f"0. 0. 0. {cell_lengths[0]:.3f} {cell_lengths[1]:.3f} {cell_lengths[2]:.3f}"

        inp_content = f"""
# Packmol input generated by Python script
tolerance {tolerance}
filetype xyz
output {output_xyz}
seed {seed}

# Host structure (Fixed)
structure {host_xyz}
  number 1
  fixed 0. 0. 0. 0. 0. 0.
end structure

# Solvent molecules
structure {solvent_xyz}
  number {n_molecules}
  inside box {box_str}
end structure
"""
        
        with open(inp_file, 'w') as f:
            f.write(inp_content)

        if verbose:
            print("入力ファイルを作成しました。Packmolを実行中...")

        # 6. Packmolの実行
        with open(inp_file, 'r') as f_in:
            result = subprocess.run(
                packmol_path, 
                stdin=f_in, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                cwd=work_dir
            )

        # 7. 結果の確認と読み込み
        if result.returncode != 0:
            print("Error: Packmol execution failed.")
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError("Packmolの実行に失敗しました。")
        
        if "Success!" in result.stdout or os.path.exists(output_xyz):
            if verbose:
                print("Packmol実行成功。構造を読み込んでいます...")
            
            # 生成された構造を読み込む
            packed_atoms = read(output_xyz)
            
            # 整形後の直方体セル情報を適用
            packed_atoms.set_cell(working_host.get_cell())
            packed_atoms.set_pbc(working_host.get_pbc())
            
            if verbose:
                print(f"生成された構造: {packed_atoms.get_chemical_formula()}")
            
            return packed_atoms
            
        else:
            print("Warning: Packmol finished but output file might be missing or incomplete.")
            print(result.stdout)
            raise RuntimeError("出力ファイルが生成されませんでした。")

# --- 使用例 ---
if __name__ == "__main__":
    from ase.build import bulk
    
    print("Packmol連携テストを実行します...")
    
    # テスト: 斜交セルを持つ構造を作成 (Al FCCを歪ませる)
    host = bulk('Al', 'fcc', a=4.05).repeat((2, 2, 2))
    # 意図的に斜めにする (Triclinic化)
    cell = host.get_cell()
    cell[1, 0] = 3.0 # Shearを大きくする
    host.set_cell(cell, scale_atoms=True)
    
    # 空隙を作る（テスト用）
    del host[[atom.index for atom in host if atom.index % 3 == 0]]

    print(f"Host (Original): {host.cell.cellpar().round(3)}")
    
    if check_packmol_command():
        try:
            packed_structure = fill_box_with_packmol(
                host_atoms=host,
                solvent_type='H2O',
                density_g_cm3=0.8,
                tolerance=2.0,
                verbose=True
            )
            write('packmol_ortho_result.xyz', packed_structure)
            print("\n正常に終了しました。")
        except Exception as e:
            print(f"\nエラー: {e}")