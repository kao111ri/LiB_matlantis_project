"""
PVDFアモルファス作成プログラム

目的:
  - ランダムに配置したPVDF鎖をNPT計算(高温圧縮→冷却)にかける
  - 実験値に近い密度(~1.78 g/cm³)の樹脂モデルを構築
  - 構造緩和と密度最適化を自動化

使用方法:
  1. PVDF分子鎖の構造ファイル(XYZ, CIF等)を準備
  2. CONFIG内のパラメータを調整
  3. このスクリプトを実行
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, List

from ase import Atoms
from ase.io import read, write
from ase import units

# プロジェクトのutilsをインポート
sys.path.append(str(Path(__file__).parent.parent / "LiB2_structure_ipynb"))
from utils.io_utils import generate_output_filename_prefix

# Matlantis関連のインポート
try:
    from matlantis_features.features.md import (
        MDFeature,
        ASEMDSystem,
        MDExtensionBase,
        NPTIntegrator,
    )
    from matlantis_features.utils.calculators import pfp_estimator_fn

    # pfcc_extras (LiquidGenerator)
    try:
        from pfcc_extras.liquidgenerator.liquid_generator import LiquidGenerator
        LIQUIDGENERATOR_AVAILABLE = True
    except ImportError:
        print("警告: pfcc_extras.liquidgenerator が見つかりません")
        LIQUIDGENERATOR_AVAILABLE = False

    MATLANTIS_AVAILABLE = True
except ImportError:
    print("警告: matlantis_features が見つかりません")
    MATLANTIS_AVAILABLE = False
    LIQUIDGENERATOR_AVAILABLE = False

# ========================================================================
# 設定パラメータ
# ========================================================================

CONFIG = {
    # PVDF分子鎖の入力ファイル
    'pvdf_chain_file': 'pvdfchain.gjf',  # または .xyz, .cif など

    # システム構築パラメータ
    'n_molecules': 30,              # PVDF分子鎖の数
    'initial_density': 0.8,         # 初期充填密度 (g/cm³) - 最終密度より低く設定
    'target_density': 1.78,         # 目標密度 (g/cm³) - PVDF実験値

    # LiquidGenerator設定
    'lg_epochs': 100,               # LiquidGeneratorの最適化エポック数

    # NPT圧縮計算パラメータ
    'compress_temp': 473.0,         # 圧縮温度 (K) - 高温で緩和促進
    'compress_pressure': 0.1,       # 圧縮圧力 (GPa)
    'compress_time': 75.0,          # 圧縮時間 (ps)
    'timestep': 1.0,                # タイムステップ (fs)

    # 冷却計算パラメータ (オプション)
    'cooling_enabled': True,        # 冷却計算を実行するか
    'cooling_temp_start': 473.0,    # 冷却開始温度 (K)
    'cooling_temp_end': 300.0,      # 冷却終了温度 (K)
    'cooling_time': 50.0,           # 冷却時間 (ps)
    'cooling_pressure': 0.0001,     # 冷却時の圧力 (GPa) - ほぼ0

    # Matlantis設定
    'model_version': 'v7.0.0',
    'calc_mode': 'CRYSTAL_U0_PLUS_D3',  # 分子系にはD3補正が推奨

    # 出力ディレクトリ
    'output_dir': 'pvdf_amorphous_results',
}

# ========================================================================
# カスタムロガークラス
# ========================================================================

class NPTLogger(MDExtensionBase):
    """NPT計算のログを記録するクラス"""

    def __init__(self, fname: str, dirout: str = '.', stdout: bool = True):
        self.fname = fname
        self.dirout = dirout
        self.t_start = perf_counter()
        self.stdout = stdout

    def __call__(self, system, integrator):
        n_step = system.current_total_step
        sim_time_ps = system.current_total_time / 1000.0
        atoms = system.ase_atoms

        E_pot = atoms.get_potential_energy()
        temp = atoms.get_temperature()

        # 密度計算
        try:
            mass = atoms.get_masses().sum()
            volume = atoms.cell.volume
            density = (mass / units.mol) / (volume * 1e-24)  # g/cm³
        except:
            density = 0.0

        # セルサイズ
        cell_params = atoms.cell.cellpar()
        a, b, c = cell_params[:3]

        calc_time = (perf_counter() - self.t_start) / 60.

        if n_step == 0:
            hdr = 'step,time[ps],E_pot[eV],T[K],density[g/cm3],a[A],b[A],c[A],calc_time[min]'
            with open(f'{self.dirout}/{self.fname}.log', 'w') as f:
                f.write(f'{hdr}\n')

        line = (f'{n_step:8d},{sim_time_ps:7.2f},{E_pot:11.4f},{temp:8.2f},'
                f'{density:7.3f},{a:8.3f},{b:8.3f},{c:8.3f},{calc_time:8.2f}')

        with open(f'{self.dirout}/{self.fname}.log', 'a') as f:
            f.write(f'{line}\n')

        if self.stdout and n_step % 500 == 0:
            print(f"NPT: t={sim_time_ps:.2f}ps T={temp:.1f}K ρ={density:.3f}g/cm³ "
                  f"cell=({a:.2f},{b:.2f},{c:.2f})Å")


# ========================================================================
# 構造構築関数
# ========================================================================

def build_initial_structure(config: Dict) -> Atoms:
    """
    LiquidGeneratorを使ってPVDF分子の初期配置を生成する

    Args:
        config: 設定パラメータ

    Returns:
        初期構造のAtomsオブジェクト
    """
    print("\n=== STEP 1: 初期構造生成 ===\n")

    # PVDF分子鎖の読み込み
    pvdf_file = config['pvdf_chain_file']

    if not os.path.exists(pvdf_file):
        print(f"✗ エラー: PVDF分子ファイルが見つかりません: {pvdf_file}")
        print("\n代替手段: 簡易的なPVDF鎖を生成します")

        # 簡易的なPVDF鎖 (C2H2F2)n の構築
        pvdf_chain = Atoms()
        n_units = 10  # 10ユニット

        for i in range(n_units):
            x = i * 2.5
            # C-C-H-H-F-F
            pvdf_chain.extend(Atoms('C2H2F2', positions=[
                [x, 0, 0],
                [x + 1.5, 0, 0],
                [x + 0.5, 1.0, 0],
                [x + 2.0, 1.0, 0],
                [x, -1.0, 0.5],
                [x + 1.5, -1.0, 0.5]
            ]))

        atoms_pvdf = pvdf_chain
        print(f"  簡易PVDF鎖を生成しました: {len(atoms_pvdf)} 原子")

    else:
        try:
            atoms_pvdf = read(pvdf_file)
            print(f"✓ PVDF分子鎖を読み込みました: {pvdf_file}")
            print(f"  原子数: {len(atoms_pvdf)}")
        except Exception as e:
            print(f"✗ エラー: ファイルの読み込みに失敗: {e}")
            return Atoms()

    # 分子の質量を計算
    n_molecules = config['n_molecules']
    mass_per_molecule = sum(atoms_pvdf.get_masses())
    total_mass = mass_per_molecule * n_molecules

    # 初期密度からセルサイズを計算
    initial_density = config['initial_density']
    volume_A3 = (total_mass / units.mol) / (initial_density / 1e24)
    cell_side_length = volume_A3 ** (1/3)

    print(f"\n  分子数: {n_molecules}")
    print(f"  総質量: {total_mass:.2f} amu")
    print(f"  初期密度: {initial_density} g/cm³")
    print(f"  セルサイズ: {cell_side_length:.2f} Å")

    # LiquidGeneratorで構造生成
    if LIQUIDGENERATOR_AVAILABLE:
        print(f"\n=== STEP 2: LiquidGenerator による配置最適化 ===\n")

        params = {
            "cell": [[cell_side_length, 0, 0], [0, cell_side_length, 0], [0, 0, cell_side_length]],
            "composition": [atoms_pvdf] * n_molecules,
            "epochs": config['lg_epochs']
        }

        try:
            generator = LiquidGenerator("torch", **params)
            atoms = generator.run()
            atoms.pbc = True

            print(f"\n✓ LiquidGenerator完了")
            print(f"  生成された原子数: {len(atoms)}")
            print(f"  セルサイズ: {atoms.cell.cellpar()[:3]}")

        except Exception as e:
            print(f"✗ LiquidGeneratorエラー: {e}")
            print("  代替手段: ランダム配置を使用します")
            atoms = random_placement_fallback(atoms_pvdf, n_molecules, cell_side_length)

    else:
        print("\n  LiquidGeneratorが利用できません。ランダム配置を使用します。")
        atoms = random_placement_fallback(atoms_pvdf, n_molecules, cell_side_length)

    return atoms


def random_placement_fallback(molecule: Atoms, n_molecules: int, cell_size: float) -> Atoms:
    """
    LiquidGeneratorが使えない場合のフォールバック: ランダム配置

    Args:
        molecule: 単一分子のAtomsオブジェクト
        n_molecules: 配置する分子数
        cell_size: セルサイズ

    Returns:
        ランダム配置された系
    """
    system = Atoms(cell=[cell_size, cell_size, cell_size], pbc=True)

    for i in range(n_molecules):
        mol = molecule.copy()
        # ランダムな位置
        pos = np.random.rand(3) * cell_size
        mol.translate(pos)
        # ランダムな回転
        mol.rotate(np.random.rand() * 360, 'x')
        mol.rotate(np.random.rand() * 360, 'y')
        mol.rotate(np.random.rand() * 360, 'z')
        system.extend(mol)

    print(f"  ランダム配置完了: {len(system)} 原子")
    return system


# ========================================================================
# NPT圧縮計算
# ========================================================================

def run_npt_compression(atoms: Atoms, config: Dict, file_prefix: str = "") -> Atoms:
    """
    NPT計算で高温圧縮を実行し、密度を上げる

    Args:
        atoms: 初期構造
        config: 設定パラメータ
        file_prefix: ファイル名プレフィックス（入力ファイル名由来）

    Returns:
        圧縮後の構造
    """
    if not MATLANTIS_AVAILABLE:
        print("✗ Matlantis環境が利用できないため、圧縮計算をスキップします")
        return atoms

    print(f"\n=== STEP 3: NPT圧縮計算 ===\n")

    output_dir = config['output_dir']
    fname = f"{file_prefix}_pvdf_npt_compression" if file_prefix else "pvdf_npt_compression"

    # 計算ステップ数
    n_steps = int(config['compress_time'] * 1000 / config['timestep'])

    # Estimator
    estimator_fn = pfp_estimator_fn(
        model_version=config['model_version'],
        calc_mode=config['calc_mode']
    )

    # System
    system = ASEMDSystem(atoms.copy(), step=0, time=0.0)
    system.init_temperature(config['compress_temp'])

    # Integrator (NPT)
    integrator = NPTIntegrator(
        timestep=config['timestep'],
        temperature=config['compress_temp'],
        pressure=config['compress_pressure'] * units.GPa,
        ttime=20,
        pfactor=2e6 * units.GPa * (units.fs**2),
        mask=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 全方向に圧縮
    )

    # Feature
    md = MDFeature(
        integrator=integrator,
        n_run=n_steps,
        traj_file_name=f"{output_dir}/{fname}.traj",
        traj_freq=100,
        estimator_fn=estimator_fn,
        logger_interval=100,
        show_logger=False,
        show_progress_bar=True
    )

    # Logger
    logger = NPTLogger(fname, dirout=output_dir, stdout=True)

    print(f"  温度: {config['compress_temp']} K")
    print(f"  圧力: {config['compress_pressure']} GPa")
    print(f"  時間: {config['compress_time']} ps ({n_steps} steps)\n")

    # 実行
    md(system, extensions=[(logger, 100)])

    # 最終構造
    final_atoms = system.ase_atoms.copy()

    # 密度確認
    mass = final_atoms.get_masses().sum()
    volume = final_atoms.cell.volume
    final_density = (mass / units.mol) / (volume * 1e-24)

    print(f"\n✓ 圧縮計算完了")
    print(f"  最終密度: {final_density:.3f} g/cm³ (目標: {config['target_density']} g/cm³)")
    print(f"  セルサイズ: {final_atoms.cell.cellpar()[:3]}\n")

    return final_atoms


# ========================================================================
# 冷却計算 (オプション)
# ========================================================================

def run_npt_cooling(atoms: Atoms, config: Dict, file_prefix: str = "") -> Atoms:
    """
    NPT計算で冷却を実行し、室温で緩和させる

    Args:
        atoms: 圧縮後の構造
        config: 設定パラメータ
        file_prefix: ファイル名プレフィックス（入力ファイル名由来）

    Returns:
        冷却後の構造
    """
    if not config['cooling_enabled'] or not MATLANTIS_AVAILABLE:
        return atoms

    print(f"\n=== STEP 4: NPT冷却計算 ===\n")

    output_dir = config['output_dir']
    fname = f"{file_prefix}_pvdf_npt_cooling" if file_prefix else "pvdf_npt_cooling"

    n_steps = int(config['cooling_time'] * 1000 / config['timestep'])

    estimator_fn = pfp_estimator_fn(
        model_version=config['model_version'],
        calc_mode=config['calc_mode']
    )

    # 温度スケジューラ (線形冷却)
    # 注: TemperatureSchedulerが使えない場合は、段階的に温度を下げる方法もあります
    # ここでは簡易的に終了温度で一定とします

    system = ASEMDSystem(atoms.copy(), step=0, time=0.0)
    system.init_temperature(config['cooling_temp_end'])

    integrator = NPTIntegrator(
        timestep=config['timestep'],
        temperature=config['cooling_temp_end'],
        pressure=config['cooling_pressure'] * units.GPa,
        ttime=20,
        pfactor=2e6 * units.GPa * (units.fs**2),
        mask=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    )

    md = MDFeature(
        integrator=integrator,
        n_run=n_steps,
        traj_file_name=f"{output_dir}/{fname}.traj",
        traj_freq=100,
        estimator_fn=estimator_fn,
        logger_interval=100,
        show_logger=False,
        show_progress_bar=True
    )

    logger = NPTLogger(fname, dirout=output_dir, stdout=True)

    print(f"  温度: {config['cooling_temp_end']} K")
    print(f"  圧力: {config['cooling_pressure']} GPa")
    print(f"  時間: {config['cooling_time']} ps\n")

    md(system, extensions=[(logger, 100)])

    final_atoms = system.ase_atoms.copy()

    mass = final_atoms.get_masses().sum()
    volume = final_atoms.cell.volume
    final_density = (mass / units.mol) / (volume * 1e-24)

    print(f"\n✓ 冷却計算完了")
    print(f"  最終密度: {final_density:.3f} g/cm³")
    print(f"  セルサイズ: {final_atoms.cell.cellpar()[:3]}\n")

    return final_atoms


# ========================================================================
# 結果プロット
# ========================================================================

def plot_density_evolution(config: Dict, file_prefix: str = ""):
    """
    密度の時間変化をプロットする

    Args:
        config: 設定パラメータ
        file_prefix: ファイル名プレフィックス（入力ファイル名由来）
    """
    print("\n=== 結果プロット ===\n")

    output_dir = Path(config['output_dir'])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # 圧縮計算
    compress_log_filename = f"{file_prefix}_pvdf_npt_compression.log" if file_prefix else "pvdf_npt_compression.log"
    compress_log = output_dir / compress_log_filename
    if compress_log.exists():
        df = pd.read_csv(compress_log)
        axes[0].plot(df['time[ps]'], df['density[g/cm3]'], '-', color='#E63946', linewidth=2)
        axes[0].axhline(y=config['target_density'], color='gray', linestyle='--', label='Target')
        axes[0].set_xlabel('Time (ps)', fontsize=12)
        axes[0].set_ylabel('Density (g/cm³)', fontsize=12)
        axes[0].set_title('NPT Compression', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # 冷却計算
    cooling_log_filename = f"{file_prefix}_pvdf_npt_cooling.log" if file_prefix else "pvdf_npt_cooling.log"
    cooling_log = output_dir / cooling_log_filename
    if cooling_log.exists():
        df = pd.read_csv(cooling_log)
        axes[1].plot(df['time[ps]'], df['density[g/cm3]'], '-', color='#457B9D', linewidth=2)
        axes[1].axhline(y=config['target_density'], color='gray', linestyle='--', label='Target')
        axes[1].set_xlabel('Time (ps)', fontsize=12)
        axes[1].set_ylabel('Density (g/cm³)', fontsize=12)
        axes[1].set_title('NPT Cooling', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # ファイル名にプレフィックスを追加
    plot_filename = f"{file_prefix}_density_evolution.png" if file_prefix else "density_evolution.png"
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ グラフを保存しました: {plot_path}\n")


# ========================================================================
# メイン実行部
# ========================================================================

def main():
    """メイン実行関数"""

    print("\n" + "=" * 60)
    print("  PVDFアモルファス作成プログラム")
    print("=" * 60 + "\n")

    # 出力ディレクトリの作成（共通）
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)

    # 入力ファイル名に基づいてファイル名プレフィックスを生成
    file_prefix = generate_output_filename_prefix(CONFIG.get('pvdf_chain_file'))

    print(f"出力ディレクトリ: {output_dir}")
    if file_prefix:
        print(f"ファイル名プレフィックス: {file_prefix}")
    print()

    # STEP 1-2: 初期構造生成
    atoms_initial = build_initial_structure(CONFIG)

    if len(atoms_initial) == 0:
        print("\n✗ 初期構造の生成に失敗しました")
        return

    # 初期構造の保存（ファイル名にプレフィックスを追加）
    init_filename = f"{file_prefix}_pvdf_initial.xyz" if file_prefix else "pvdf_initial.xyz"
    init_path = output_dir / init_filename
    write(init_path, atoms_initial)
    print(f"✓ 初期構造を保存しました: {init_path}\n")

    if not MATLANTIS_AVAILABLE:
        print("\n" + "=" * 60)
        print("  Matlantis環境が利用できません")
        print("=" * 60)
        print("\n初期構造のみ保存しました。")
        print("NPT計算を実行するには、Matlantis環境が必要です。\n")
        return

    # STEP 3: NPT圧縮
    atoms_compressed = run_npt_compression(atoms_initial, CONFIG, file_prefix)

    # 圧縮後の構造を保存（ファイル名にプレフィックスを追加）
    compressed_filename = f"{file_prefix}_pvdf_compressed.xyz" if file_prefix else "pvdf_compressed.xyz"
    compressed_path = output_dir / compressed_filename
    write(compressed_path, atoms_compressed)
    print(f"✓ 圧縮後の構造を保存しました: {compressed_path}\n")

    # STEP 4: NPT冷却 (オプション)
    atoms_final = run_npt_cooling(atoms_compressed, CONFIG, file_prefix)

    # 最終構造の保存（ファイル名にプレフィックスを追加）
    final_filename = f"{file_prefix}_pvdf_final_amorphous.xyz" if file_prefix else "pvdf_final_amorphous.xyz"
    final_path = output_dir / final_filename
    write(final_path, atoms_final)
    print(f"✓ 最終構造を保存しました: {final_path}\n")

    # 結果プロット
    plot_density_evolution(CONFIG, file_prefix)

    print("\n" + "=" * 60)
    print("  PVDF アモルファス作成 完了")
    print("=" * 60)
    print("\n【確認事項】")
    print("1. 最終密度が目標値(~1.78 g/cm³)に近いか")
    print("2. セル形状が大きく歪んでいないか")
    print("3. 分子が適切に緩和されているか")
    print("\n【次のステップ】")
    print("1. 最終構造をビューアで確認")
    print("2. 必要に応じてパラメータを調整して再実行")
    print("3. この構造を界面モデル構築に使用\n")


if __name__ == "__main__":
    main()
