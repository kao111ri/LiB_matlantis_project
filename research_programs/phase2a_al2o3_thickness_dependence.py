"""
Phase 2-A: Al₂O₃層厚依存性検証プログラム

目的:
  - Al₂O₃層の厚さ(層数)を変えたNMC/Al₂O₃/PVDF系を構築
  - 段階的加熱MD(LiPF₆分解→PVDF分解)を実行
  - AlF₃生成量とF貫通性の層厚依存性を評価

確認事項:
  1. 臨界層厚(剥離性が急落する厚さ)の特定
  2. F貫通の活性化障壁
  3. 予想: 3層(~1.5nm)以上で急減
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple

from ase import Atoms
from ase.io import read, write
from ase.build import surface, add_adsorbate, molecule
from ase.constraints import FixAtoms
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
        NVTBerendsenIntegrator,
    )
    from matlantis_features.utils.calculators import pfp_estimator_fn
    from matlantis_features.features.md.md_extensions import TemperatureScheduler
    MATLANTIS_AVAILABLE = True
except ImportError:
    print("警告: matlantis_features が見つかりません")
    MATLANTIS_AVAILABLE = False

# ========================================================================
# 設定パラメータ
# ========================================================================

CONFIG = {
    # Al₂O₃層数のバリエーション
    'al2o3_layers': [1, 2, 3, 5],  # テストする層数

    # 構造パラメータ
    'al_metal_thickness': 10.0,   # Al金属層の厚さ (Å)
    'pvdf_thickness': 30.0,        # PVDF層の厚さ (Å)
    'cell_lateral_size': 15.0,     # X-Y方向のセルサイズ (Å)

    # 分子数
    'n_lipf6': 10,                 # LiPF₆分子数
    'n_h2o': 5,                    # H₂O分子数

    # MD計算パラメータ
    'stage1_temp': 400.0,          # Stage1温度: LiPF₆分解 (K)
    'stage1_time': 20.0,           # Stage1時間 (ps)
    'stage2_temp': 800.0,          # Stage2温度: PVDF分解 (K)
    'stage2_time': 50.0,           # Stage2時間 (ps)
    'timestep': 0.5,               # タイムステップ (fs)

    # Matlantis設定
    'model_version': 'v7.0.0',
    'calc_mode': 'CRYSTAL_U0',

    # 出力ディレクトリ
    'output_dir': 'phase2a_al2o3_thickness_results',
}

# ========================================================================
# カスタムロガークラス
# ========================================================================

class PrintWriteLog(MDExtensionBase):
    """標準的なMDログを記録するクラス"""

    def __init__(self, fname: str, dirout: str = '.', stdout: bool = False):
        self.fname = fname
        self.dirout = dirout
        self.t_start = perf_counter()
        self.stdout = stdout

    def __call__(self, system, integrator):
        n_step = system.current_total_step
        sim_time_ps = system.current_total_time / 1000.0
        E_pot = system.ase_atoms.get_potential_energy()
        temp = system.ase_atoms.get_temperature()
        calc_time = (perf_counter() - self.t_start) / 60.

        if n_step == 0:
            with open(f'{self.dirout}/{self.fname}.log', 'w') as f:
                f.write('step,time[ps],E_pot[eV],T[K],calc_time[min]\n')

        line = f'{n_step:8d},{sim_time_ps:7.2f},{E_pot:11.4f},{temp:8.2f},{calc_time:8.2f}'

        with open(f'{self.dirout}/{self.fname}.log', 'a') as f:
            f.write(f'{line}\n')

        if self.stdout and n_step % 1000 == 0:
            print(f"MD: t={sim_time_ps:.2f}ps T={temp:.1f}K E={E_pot:.2f}eV")


class TrackInterfaceReaction(MDExtensionBase):
    """Al₂O₃/PVDF界面での反応を追跡するクラス"""

    def __init__(self, fname: str, dirout: str = '.', stdout: bool = True):
        self.fname = fname
        self.dirout = dirout
        self.stdout = stdout
        self.log_path = f'{self.dirout}/{self.fname}_interface.log'

    def __call__(self, system, integrator):
        n_step = system.current_total_step
        sim_time_ps = system.current_total_time / 1000.0
        atoms = system.ase_atoms

        if n_step == 0:
            with open(self.log_path, 'w') as f:
                f.write('step,time[ps],n_AlF,n_HF,F_penetration_A\n')

        # 原子インデックス
        al_idx = [a.index for a in atoms if a.symbol == 'Al']
        f_idx = [a.index for a in atoms if a.symbol == 'F']
        h_idx = [a.index for a in atoms if a.symbol == 'H']

        n_alf = n_hf = 0
        f_penetration = 0.0

        try:
            all_dists = atoms.get_all_distances(mic=True)

            # Al-F結合
            if al_idx and f_idx:
                alf_dists = all_dists[np.ix_(al_idx, f_idx)]
                n_alf = (alf_dists < 2.0).sum()

            # HF結合
            if h_idx and f_idx:
                hf_dists = all_dists[np.ix_(h_idx, f_idx)]
                n_hf = (hf_dists < 1.0).sum()

            # F貫通深度 (Al₂O₃表面からの侵入)
            if f_idx and al_idx:
                f_z = atoms.positions[f_idx][:, 2]
                al_z = atoms.positions[al_idx][:, 2]
                # Al₂O₃表面の定義（Al原子の上位20%）
                al_surface_z = np.percentile(al_z, 80)
                penetrating = f_z[f_z < al_surface_z]
                if len(penetrating) > 0:
                    f_penetration = al_surface_z - penetrating.min()

        except Exception as e:
            print(f"Interface tracking error: {e}")

        line = f'{n_step:8d},{sim_time_ps:7.2f},{n_alf:5d},{n_hf:5d},{f_penetration:7.2f}'

        with open(self.log_path, 'a') as f:
            f.write(f'{line}\n')

        if self.stdout and n_step % 1000 == 0:
            print(f"INTERFACE: AlF={n_alf} HF={n_hf} F_pen={f_penetration:.2f}A")


# ========================================================================
# 構造構築関数
# ========================================================================

def build_layered_system(n_al2o3_layers: int, config: Dict) -> Atoms:
    """
    Al金属/Al₂O₃/PVDF の多層構造を構築する

    Args:
        n_al2o3_layers: Al₂O₃の層数
        config: 設定パラメータ

    Returns:
        構築された系
    """
    print(f"\n=== 構造構築: Al₂O₃ {n_al2o3_layers}層 ===\n")

    cell_size = config['cell_lateral_size']
    system = Atoms(cell=[cell_size, cell_size, 100.0], pbc=True)  # Z方向は後で調整

    z_current = 5.0  # 開始位置

    # 1. Al金属層 (底面, 固定)
    print("  Al金属層を構築中...")
    al_lattice = 4.05  # Al格子定数
    n_al_layers = int(config['al_metal_thickness'] / (al_lattice / 2))

    for i in range(n_al_layers):
        z = z_current + i * al_lattice / 2
        # 簡易的な2Dグリッド配置
        n_grid = int(cell_size / al_lattice)
        for ix in range(n_grid):
            for iy in range(n_grid):
                x = ix * al_lattice + (i % 2) * al_lattice / 2
                y = iy * al_lattice + (i % 2) * al_lattice / 2
                if x < cell_size and y < cell_size:
                    system.append(Atoms('Al', positions=[[x, y, z]], tags=[1]))

    z_current += n_al_layers * al_lattice / 2
    print(f"    Al金属層: {len([a for a in system if a.tag == 1])} 原子")

    # 2. Al₂O₃層
    print("  Al₂O₃層を構築中...")
    # 簡易的なAl₂O₃構造（実際にはα-Al₂O₃結晶構造を使用することを推奨）
    al2o3_layer_thickness = 0.5  # 1層あたり約0.5 Å

    for layer in range(n_al2o3_layers):
        z_layer = z_current + layer * al2o3_layer_thickness

        # Al原子配置
        n_grid = int(cell_size / 3.0)
        for ix in range(n_grid):
            for iy in range(n_grid):
                x = ix * 3.0
                y = iy * 3.0
                system.append(Atoms('Al', positions=[[x, y, z_layer]], tags=[2]))

                # O原子を周囲に配置
                for dx, dy in [(1, 0), (-1, 0), (0, 1)]:
                    ox = x + dx * 1.5
                    oy = y + dy * 1.5
                    if 0 <= ox < cell_size and 0 <= oy < cell_size:
                        system.append(Atoms('O', positions=[[ox, oy, z_layer + 0.2]], tags=[2]))

    z_current += n_al2o3_layers * al2o3_layer_thickness
    print(f"    Al₂O₃層: {len([a for a in system if a.tag == 2 and a.symbol == 'Al'])} Al, "
          f"{len([a for a in system if a.tag == 2 and a.symbol == 'O'])} O")

    # 3. PVDF層 (簡易版)
    print("  PVDF層を構築中...")
    # 実際にはPVDF分子構造を使用、ここでは簡易的にC, H, Fをランダム配置
    pvdf_z_range = (z_current + 2, z_current + config['pvdf_thickness'])

    # PVDFの化学式: (C2H2F2)n
    n_pvdf_units = 20
    for i in range(n_pvdf_units):
        x = np.random.rand() * cell_size
        y = np.random.rand() * cell_size
        z = pvdf_z_range[0] + np.random.rand() * (pvdf_z_range[1] - pvdf_z_range[0])

        # C-C-H-H-F-F ユニット
        system.extend(Atoms('C2H2F2', positions=[
            [x, y, z],
            [x + 1.5, y, z],
            [x + 0.5, y + 1.0, z],
            [x + 2.0, y + 1.0, z],
            [x, y - 1.0, z],
            [x + 1.5, y - 1.0, z]
        ], tags=[3] * 6))

    z_current = pvdf_z_range[1] + 5.0
    print(f"    PVDF層: {len([a for a in system if a.tag == 3])} 原子")

    # 4. LiPF₆とH₂Oの追加 (PVDF層内)
    print("  LiPF₆とH₂Oを追加中...")
    for i in range(config['n_lipf6']):
        x = np.random.rand() * cell_size
        y = np.random.rand() * cell_size
        z = pvdf_z_range[0] + np.random.rand() * (pvdf_z_range[1] - pvdf_z_range[0])
        # 簡易的なLiPF₆配置
        system.append(Atoms('Li', positions=[[x, y, z]]))
        system.append(Atoms('P', positions=[[x + 1.0, y, z]]))
        for dx, dy, dz in [(0.8, 0, 0), (-0.8, 0, 0), (0, 0.8, 0), (0, -0.8, 0), (0, 0, 0.8), (0, 0, -0.8)]:
            system.append(Atoms('F', positions=[[x + 1.0 + dx, y + dy, z + dz]]))

    h2o = molecule('H2O')
    for i in range(config['n_h2o']):
        h2o_copy = h2o.copy()
        x = np.random.rand() * cell_size
        y = np.random.rand() * cell_size
        z = pvdf_z_range[0] + np.random.rand() * (pvdf_z_range[1] - pvdf_z_range[0])
        h2o_copy.translate([x, y, z])
        system.extend(h2o_copy)

    # セルサイズの最終調整
    system.set_cell([cell_size, cell_size, z_current], scale_atoms=False)

    # Al金属層の固定
    fix_indices = [a.index for a in system if a.tag == 1]
    system.set_constraint(FixAtoms(indices=fix_indices))

    print(f"\n  ✓ 構造構築完了")
    print(f"    総原子数: {len(system)}")
    print(f"    セルサイズ: {cell_size:.1f} x {cell_size:.1f} x {z_current:.1f} Å")
    print(f"    Al₂O₃層厚: ~{n_al2o3_layers * 0.5:.2f} Å\n")

    return system


# ========================================================================
# 段階加熱MD実行関数
# ========================================================================

def run_staged_heating_md(atoms: Atoms, n_layers: int, config: Dict, file_prefix: str = ""):
    """
    2段階加熱MDを実行する (Stage1: LiPF₆分解, Stage2: PVDF分解)

    Args:
        atoms: 初期構造
        n_layers: Al₂O₃の層数
        config: 設定パラメータ
        file_prefix: ファイル名プレフィックス（入力ファイル名由来）
    """
    if not MATLANTIS_AVAILABLE:
        print("✗ Matlantis環境が利用できないため、シミュレーションをスキップします")
        return

    print(f"\n=== 段階加熱MD開始: {n_layers}層 ===\n")

    output_dir = config['output_dir']
    prefix_part = f"{file_prefix}_" if file_prefix else ""
    fname_base = f"{prefix_part}al2o3_{n_layers}layers"

    estimator_fn = pfp_estimator_fn(
        model_version=config['model_version'],
        calc_mode=config['calc_mode']
    )

    # Stage 1: LiPF₆分解 (400K)
    print(f"--- Stage 1: LiPF₆分解 ({config['stage1_temp']} K, {config['stage1_time']} ps) ---\n")

    fname_s1 = f"{fname_base}_stage1"
    n_steps_s1 = int(config['stage1_time'] * 1000 / config['timestep'])

    system = ASEMDSystem(atoms.copy(), step=0, time=0.0)

    integrator_s1 = NVTBerendsenIntegrator(
        timestep=config['timestep'],
        temperature=config['stage1_temp'],
        taut=100.0,
        fixcm=True
    )

    md_s1 = MDFeature(
        integrator=integrator_s1,
        n_run=n_steps_s1,
        traj_file_name=f"{output_dir}/{fname_s1}.traj",
        traj_freq=200,
        estimator_fn=estimator_fn,
        logger_interval=200,
        show_logger=False,
        show_progress_bar=True
    )

    logger_s1 = PrintWriteLog(fname_s1, dirout=output_dir, stdout=True)
    tracker_s1 = TrackInterfaceReaction(fname_s1, dirout=output_dir, stdout=True)

    md_s1(system, extensions=[(logger_s1, 200), (tracker_s1, 200)])

    # Stage1の最終構造を取得
    atoms_after_s1 = system.ase_atoms.copy()

    print(f"\n✓ Stage 1 完了\n")

    # Stage 2: PVDF分解 (800K)
    print(f"--- Stage 2: PVDF分解 ({config['stage2_temp']} K, {config['stage2_time']} ps) ---\n")

    fname_s2 = f"{fname_base}_stage2"
    n_steps_s2 = int(config['stage2_time'] * 1000 / config['timestep'])

    system2 = ASEMDSystem(atoms_after_s1, step=0, time=0.0)

    integrator_s2 = NVTBerendsenIntegrator(
        timestep=config['timestep'],
        temperature=config['stage2_temp'],
        taut=100.0,
        fixcm=True
    )

    md_s2 = MDFeature(
        integrator=integrator_s2,
        n_run=n_steps_s2,
        traj_file_name=f"{output_dir}/{fname_s2}.traj",
        traj_freq=200,
        estimator_fn=estimator_fn,
        logger_interval=200,
        show_logger=False,
        show_progress_bar=True
    )

    logger_s2 = PrintWriteLog(fname_s2, dirout=output_dir, stdout=True)
    tracker_s2 = TrackInterfaceReaction(fname_s2, dirout=output_dir, stdout=True)

    md_s2(system2, extensions=[(logger_s2, 200), (tracker_s2, 200)])

    print(f"\n✓ Stage 2 完了\n")


# ========================================================================
# 結果解析・プロット関数
# ========================================================================

def analyze_thickness_dependence(config: Dict, file_prefix: str = ""):
    """
    層厚依存性の解析とプロット

    Args:
        config: 設定パラメータ
        file_prefix: ファイル名プレフィックス（入力ファイル名由来）
    """
    print("\n=== 層厚依存性解析 ===\n")

    output_dir = Path(config['output_dir'])
    layers_list = config['al2o3_layers']

    results = []

    for n_layers in layers_list:
        # Stage2の結果を読み込み
        prefix_part = f"{file_prefix}_" if file_prefix else ""
        fname = f"{prefix_part}al2o3_{n_layers}layers_stage2"
        log_path = output_dir / f"{fname}_interface.log"

        if not log_path.exists():
            print(f"✗ ログファイルが見つかりません: {log_path}")
            continue

        df = pd.read_csv(log_path)

        # AlF₃生成量(最終値)
        final_alf = df['n_AlF'].iloc[-1]

        # F貫通深度(最大値)
        max_penetration = df['F_penetration_A'].max()

        results.append({
            'n_layers': n_layers,
            'thickness_nm': n_layers * 0.05,  # 1層=0.5Å=0.05nm
            'AlF3_bonds': final_alf,
            'F_penetration_A': max_penetration
        })

        print(f"  {n_layers}層: AlF₃={final_alf:.0f}, F貫通={max_penetration:.2f}Å")

    if not results:
        print("\n✗ 解析可能なデータがありません\n")
        return

    df_results = pd.DataFrame(results)

    # プロット
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # AlF₃生成量 vs 層厚
    axes[0].plot(df_results['thickness_nm'], df_results['AlF3_bonds'], 'o-',
                 color='#E63946', linewidth=2, markersize=8)
    axes[0].set_xlabel('Al₂O₃ Thickness (nm)', fontsize=12)
    axes[0].set_ylabel('AlF₃ Bonds at Interface', fontsize=12)
    axes[0].set_title('AlF₃ Formation vs Thickness', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # F貫通深度 vs 層厚
    axes[1].plot(df_results['thickness_nm'], df_results['F_penetration_A'], 'o-',
                 color='#457B9D', linewidth=2, markersize=8)
    axes[1].set_xlabel('Al₂O₃ Thickness (nm)', fontsize=12)
    axes[1].set_ylabel('F Penetration Depth (Å)', fontsize=12)
    axes[1].set_title('F Penetration vs Thickness', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # ファイル名にプレフィックスを追加
    plot_filename = f"{file_prefix}_thickness_dependence.png" if file_prefix else "thickness_dependence.png"
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ グラフを保存しました: {plot_path}\n")

    # 結果テーブルの保存
    csv_filename = f"{file_prefix}_thickness_dependence.csv" if file_prefix else "thickness_dependence.csv"
    csv_path = output_dir / csv_filename
    df_results.to_csv(csv_path, index=False)
    print(f"✓ 結果テーブルを保存しました: {csv_path}\n")


# ========================================================================
# メイン実行部
# ========================================================================

def main():
    """メイン実行関数"""

    print("\n" + "=" * 60)
    print("  Phase 2-A: Al₂O₃層厚依存性検証")
    print("=" * 60 + "\n")

    # 出力ディレクトリの作成（共通）
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)

    # ファイル名プレフィックスを生成（このプログラムでは特に入力ファイルがないので空）
    # 必要に応じてタイムスタンプを使用可能
    file_prefix = generate_output_filename_prefix(use_timestamp=False)

    print(f"出力ディレクトリ: {output_dir}")
    if file_prefix:
        print(f"ファイル名プレフィックス: {file_prefix}")
    print()

    if not MATLANTIS_AVAILABLE:
        print("✗ Matlantis環境が利用できません")
        print("  このスクリプトは、Matlantis環境で実行してください。\n")
        return

    # 各層数について計算を実行
    for n_layers in CONFIG['al2o3_layers']:
        # 構造構築
        system = build_layered_system(n_layers, CONFIG)

        # 初期構造の保存（ファイル名にプレフィックスを追加）
        init_filename = f"{file_prefix}_initial_al2o3_{n_layers}layers.xyz" if file_prefix else f"initial_al2o3_{n_layers}layers.xyz"
        init_path = output_dir / init_filename
        write(init_path, system)
        print(f"✓ 初期構造を保存: {init_path}\n")

        # 段階加熱MD実行
        run_staged_heating_md(system, n_layers, CONFIG, file_prefix)

    # 層厚依存性の解析
    analyze_thickness_dependence(CONFIG, file_prefix)

    print("\n" + "=" * 60)
    print("  Phase 2-A 完了")
    print("=" * 60)
    print("\n【確認事項】")
    print("1. AlF₃生成量が層厚とともに減少するか")
    print("2. F貫通深度が層厚とともに減少するか")
    print("3. 臨界層厚(急激な変化点)が存在するか")
    print("\n【予想】")
    print("- 3層(~1.5nm)以上でAlF₃生成が急減")
    print("- F貫通が抑制され、剥離性が低下\n")


if __name__ == "__main__":
    main()
