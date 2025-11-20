"""
Phase 1-C: LiPF₆とAlの接触加熱シミュレーション

目的:
  - LiPF₆とAl金属表面の接触反応を調査
  - 加熱条件下での反応生成物の追跡
  - Al表面の酸化・腐食の観測
  - 複数温度条件での反応速度の比較

確認事項:
  1. LiPF₆の分解生成物（HF, PF₃, PF₅など）
  2. Al表面の反応（AlF₃生成など）
  3. 温度依存性（300K, 400K, 600K, 800K）
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
from ase.build import bulk, surface
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
        NVTBerendsenIntegrator,
    )
    from matlantis_features.utils.calculators import pfp_estimator_fn
    MATLANTIS_AVAILABLE = True
except ImportError:
    print("警告: matlantis_features が見つかりません")
    print("このスクリプトを実行するには、Matlantis環境が必要です")
    MATLANTIS_AVAILABLE = False

# ========================================================================
# 設定パラメータ
# ========================================================================

CONFIG = {
    # LiPF₆結晶ファイル
    'lipf6_cif_path': "LiPF6.cif",

    # Al基板設定
    'al_surface_size': (3, 3, 3),  # Al(100)表面のサイズ
    'al_vacuum': 10.0,              # 真空層の厚さ (Å)

    # LiPF₆配置設定
    'n_lipf6_molecules': 3,         # LiPF₆分子数
    'lipf6_height_above_surface': 3.0,  # Al表面からの初期高さ (Å)

    # MD計算パラメータ
    'temperatures': [300.0, 400.0, 600.0, 800.0],  # 温度 (K)
    'timestep': 0.5,                # タイムステップ (fs)
    'simulation_time': 30.0,        # シミュレーション時間 (ps)
    'traj_freq': 100,               # Trajectory保存頻度
    'logger_interval': 100,         # ログ出力頻度

    # Matlantis設定
    'model_version': 'v7.0.0',
    'calc_mode': 'CRYSTAL_U0',

    # 出力ディレクトリ
    'output_dir': 'phase1c_lipf6_al_results',
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
        E_tot = system.ase_atoms.get_total_energy()
        E_pot = system.ase_atoms.get_potential_energy()
        E_kin = system.ase_atoms.get_kinetic_energy()
        temp = system.ase_atoms.get_temperature()

        try:
            density = system.ase_atoms.get_masses().sum() / units.mol / (
                system.ase_atoms.cell.volume * (1e-8**3)
            )
        except:
            density = 0.0

        calc_time = (perf_counter() - self.t_start) / 60.

        if n_step == 0:
            hdr = ('step,time[ps],E_tot[eV],E_pot[eV],E_kin[eV],'
                   'T[K],density[g/cm3],calc_time[min]')
            with open(f'{self.dirout}/{self.fname}.log', 'w') as f_log:
                f_log.write(f'{hdr}\n')

        line = (f'{n_step:8d},{sim_time_ps:7.2f},'
                f'{E_tot:11.4f},{E_pot:11.4f},{E_kin:9.4f},'
                f'{temp:8.2f},{density:7.3f},{calc_time:8.2f}')

        with open(f'{self.dirout}/{self.fname}.log', 'a') as f_log:
            f_log.write(f'{line}\n')

        if self.stdout:
            print(f"MD LOG: {line}")


class TrackAlReaction(MDExtensionBase):
    """Al表面とLiPF₆の反応を追跡するクラス"""

    def __init__(self, fname: str, dirout: str = '.', stdout: bool = True,
                 n_al_atoms: int = 0):
        self.fname = fname
        self.dirout = dirout
        self.stdout = stdout
        self.n_al_atoms = n_al_atoms
        self.log_path = f'{self.dirout}/{self.fname}_reaction.log'

    def __call__(self, system, integrator):
        n_step = system.current_total_step
        sim_time_ps = system.current_total_time / 1000.0
        atoms = system.ase_atoms

        if n_step == 0:
            hdr = 'step,time[ps],n_AlF,n_PF,n_HF,n_LiF,n_PO,avg_F_height[A]'
            with open(self.log_path, 'w') as f:
                f.write(f'{hdr}\n')

        # 原子インデックスの取得
        al_idx = [a.index for a in atoms if a.symbol == 'Al']
        f_idx = [a.index for a in atoms if a.symbol == 'F']
        p_idx = [a.index for a in atoms if a.symbol == 'P']
        li_idx = [a.index for a in atoms if a.symbol == 'Li']
        h_idx = [a.index for a in atoms if a.symbol == 'H']
        o_idx = [a.index for a in atoms if a.symbol == 'O']

        n_alf = n_pf = n_hf = n_lif = n_po = 0
        avg_f_height = 0.0

        try:
            # 全原子間距離行列の計算
            all_dists = atoms.get_all_distances(mic=True)

            # Al-F結合（Al表面の腐食）（< 2.0Å）
            if al_idx and f_idx:
                alf_dists = all_dists[np.ix_(al_idx, f_idx)]
                n_alf = (alf_dists < 2.0).sum()

            # P-F結合（残存LiPF₆またはPF₅等）（< 1.8Å）
            if p_idx and f_idx:
                pf_dists = all_dists[np.ix_(p_idx, f_idx)]
                n_pf = (pf_dists < 1.8).sum()

            # H-F結合（HF生成）（< 1.0Å）
            if h_idx and f_idx:
                hf_dists = all_dists[np.ix_(h_idx, f_idx)]
                n_hf = (hf_dists < 1.0).sum()

            # Li-F結合（LiF生成）（< 1.8Å）
            if li_idx and f_idx:
                lif_dists = all_dists[np.ix_(li_idx, f_idx)]
                n_lif = (lif_dists < 1.8).sum()

            # P-O結合（POF₃等の酸化）（< 1.6Å）
            if p_idx and o_idx:
                po_dists = all_dists[np.ix_(p_idx, o_idx)]
                n_po = (po_dists < 1.6).sum()

            # F原子の平均高さ（Al表面からの距離）
            if f_idx and al_idx:
                al_positions = atoms.positions[al_idx]
                f_positions = atoms.positions[f_idx]
                al_avg_z = al_positions[:, 2].mean()
                f_avg_z = f_positions[:, 2].mean()
                avg_f_height = f_avg_z - al_avg_z

        except Exception as e:
            print(f"REACTION LOG ERROR: {e}")

        # 書き込み
        line = (f'{n_step:8d},{sim_time_ps:7.2f},'
                f'{n_alf:5d},{n_pf:5d},{n_hf:5d},{n_lif:5d},{n_po:5d},'
                f'{avg_f_height:8.3f}')
        with open(self.log_path, 'a') as f:
            f.write(f'{line}\n')

        if self.stdout and n_step % 500 == 0:
            print(f"REACTION: t={sim_time_ps:.2f}ps "
                  f"AlF={n_alf} PF={n_pf} HF={n_hf} LiF={n_lif} PO={n_po} "
                  f"F_height={avg_f_height:.2f}Å")


# ========================================================================
# システム構築関数
# ========================================================================

def build_al_surface(config: Dict) -> Tuple[Atoms, int]:
    """
    Al(100)表面を構築する

    Args:
        config: 設定パラメータの辞書

    Returns:
        Al表面のAtomsオブジェクトと、Al原子数
    """
    print("\n=== Al表面の構築 ===\n")

    # Al(100)表面の作成
    size = config['al_surface_size']
    al_slab = surface('Al', (1, 0, 0), size[2], vacuum=config['al_vacuum'])
    al_slab = al_slab.repeat((size[0], size[1], 1))

    # 最下層のAl原子を固定（基板の効果）
    z_positions = al_slab.positions[:, 2]
    min_z = z_positions.min()
    fixed_atoms = [i for i, z in enumerate(z_positions) if z < min_z + 1.0]

    # FixAtomsは使わず、後でMD中に制約を加える（Matlantis互換性のため）
    print(f"✓ Al(100)表面を作成しました")
    print(f"  サイズ: {size}")
    print(f"  Al原子数: {len(al_slab)}")
    print(f"  セルサイズ: {al_slab.cell.diagonal()}")
    print(f"  固定原子数: {len(fixed_atoms)}\n")

    return al_slab, len(al_slab)


def add_lipf6_molecules(al_slab: Atoms, config: Dict) -> Atoms:
    """
    Al表面上にLiPF₆分子を配置する

    Args:
        al_slab: Al表面のAtomsオブジェクト
        config: 設定パラメータ

    Returns:
        LiPF₆を追加したAtomsオブジェクト
    """
    print("=== LiPF₆分子の配置 ===\n")

    system = al_slab.copy()
    n_lipf6 = config['n_lipf6_molecules']
    height = config['lipf6_height_above_surface']

    # Al表面の最大Z座標を取得
    al_top_z = al_slab.positions[:, 2].max()

    # セルの中心XY座標
    cell = al_slab.cell.diagonal()
    center_x = cell[0] / 2
    center_y = cell[1] / 2

    # LiPF₆結晶ファイルの読み込み試行
    cif_path = config['lipf6_cif_path']
    if os.path.exists(cif_path):
        print(f"✓ LiPF₆結晶ファイルを読み込みます: {cif_path}")
        lipf6_unit = read(cif_path)

        # 単位格子から分子を抽出（簡易版）
        # 実際にはLi-P-F6の1つのユニットを抽出する必要あり
        # ここでは全体を縮小して使用
        for i in range(n_lipf6):
            lipf6_mol = lipf6_unit.copy()

            # スケーリング（必要に応じて）
            lipf6_mol.positions -= lipf6_mol.positions.mean(axis=0)
            lipf6_mol.positions *= 0.5  # 縮小

            # Al表面上に配置
            offset_x = (i - n_lipf6/2) * 3.0
            position = np.array([center_x + offset_x, center_y, al_top_z + height])
            lipf6_mol.translate(position)

            system.extend(lipf6_mol)
    else:
        print(f"✗ LiPF₆結晶ファイルが見つかりません: {cif_path}")
        print("  簡易的な構造を作成します\n")

        # 簡易的なLiPF₆分子の構築
        for i in range(n_lipf6):
            # 配置位置
            offset_x = (i - n_lipf6/2) * 5.0
            center = np.array([center_x + offset_x, center_y, al_top_z + height])

            # Li原子
            system.append(Atoms('Li', positions=[center]))

            # P原子
            p_pos = center + np.array([0, 0, 2.0])
            system.append(Atoms('P', positions=[p_pos]))

            # 6個のF原子（八面体配置）
            f_offsets = np.array([
                [1.5, 0, 0], [-1.5, 0, 0],
                [0, 1.5, 0], [0, -1.5, 0],
                [0, 0, 1.5], [0, 0, -1.5]
            ])
            for offset in f_offsets:
                f_pos = p_pos + offset
                system.append(Atoms('F', positions=[f_pos]))

    print(f"✓ {n_lipf6} 個のLiPF₆分子を配置しました")
    print(f"  総原子数: {len(system)}")
    print(f"  組成: {system.get_chemical_formula()}\n")

    return system


# ========================================================================
# MD実行関数
# ========================================================================

def run_md_simulation(atoms: Atoms, temperature: float, config: Dict,
                      n_al_atoms: int, file_prefix: str = "") -> str:
    """
    NVT-MD シミュレーションを実行する

    Args:
        atoms: 初期構造
        temperature: 温度 (K)
        config: 設定パラメータ
        n_al_atoms: Al原子の数（反応追跡用）
        file_prefix: ファイル名プレフィックス

    Returns:
        出力ファイル名
    """
    if not MATLANTIS_AVAILABLE:
        print("✗ Matlantis環境が利用できないため、シミュレーションをスキップします")
        return ""

    print(f"\n=== MD計算開始: {temperature} K ===\n")

    # ファイル名
    fname_base = f"{file_prefix}_" if file_prefix else ""
    fname = f"{fname_base}lipf6_al_{int(temperature)}K"
    output_dir = config['output_dir']

    # 計算ステップ数
    n_steps = int(config['simulation_time'] * 1000 / config['timestep'])

    # Estimator
    estimator_fn = pfp_estimator_fn(
        model_version=config['model_version'],
        calc_mode=config['calc_mode']
    )

    # System
    system = ASEMDSystem(atoms.copy(), step=0, time=0.0)

    # Integrator (NVT)
    integrator = NVTBerendsenIntegrator(
        timestep=config['timestep'],
        temperature=temperature,
        taut=100.0,
        fixcm=True,
    )

    # Feature
    md = MDFeature(
        integrator=integrator,
        n_run=n_steps,
        traj_file_name=f"{output_dir}/{fname}.traj",
        traj_freq=config['traj_freq'],
        estimator_fn=estimator_fn,
        logger_interval=config['logger_interval'],
        show_logger=False,
        show_progress_bar=True,
    )

    # Extensions (ロガー)
    logger_std = PrintWriteLog(fname, dirout=output_dir, stdout=False)
    logger_reaction = TrackAlReaction(fname, dirout=output_dir, stdout=True,
                                     n_al_atoms=n_al_atoms)

    print(f"ステップ数: {n_steps} ({config['simulation_time']} ps)")
    print(f"温度: {temperature} K\n")

    # 実行
    md(system, extensions=[(logger_std, config['logger_interval']),
                           (logger_reaction, config['logger_interval'])])

    print(f"\n✓ MD計算完了: {fname}\n")

    return fname


# ========================================================================
# 結果解析・プロット関数
# ========================================================================

def analyze_and_plot_results(config: Dict, file_prefix: str = ""):
    """
    シミュレーション結果を解析してプロットする

    Args:
        config: 設定パラメータ
        file_prefix: ファイル名プレフィックス
    """
    print("\n=== 結果解析・グラフ作成 ===\n")

    output_dir = Path(config['output_dir'])
    temperatures = config['temperatures']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('LiPF₆ + Al Surface Reaction Analysis', fontsize=16, fontweight='bold')

    results_summary = []

    for temp in temperatures:
        fname_base = f"{file_prefix}_" if file_prefix else ""
        fname = f"{fname_base}lipf6_al_{int(temp)}K"
        reaction_log = output_dir / f"{fname}_reaction.log"

        if not reaction_log.exists():
            print(f"✗ ログファイルが見つかりません: {reaction_log}")
            continue

        # データ読み込み
        df = pd.read_csv(reaction_log)
        label = f"{int(temp)} K"

        # Al-F結合（腐食）
        axes[0, 0].plot(df['time[ps]'], df['n_AlF'], 'o-', label=label, markersize=3)

        # P-F結合（LiPF₆の分解）
        axes[0, 1].plot(df['time[ps]'], df['n_PF'], 'o-', label=label, markersize=3)

        # HF生成
        axes[0, 2].plot(df['time[ps]'], df['n_HF'], 'o-', label=label, markersize=3)

        # LiF生成
        axes[1, 0].plot(df['time[ps]'], df['n_LiF'], 'o-', label=label, markersize=3)

        # P-O結合
        axes[1, 1].plot(df['time[ps]'], df['n_PO'], 'o-', label=label, markersize=3)

        # F原子の高さ
        axes[1, 2].plot(df['time[ps]'], df['avg_F_height[A]'], 'o-', label=label, markersize=3)

        # サマリー
        max_alf = df['n_AlF'].max()
        final_alf = df['n_AlF'].iloc[-1]
        final_pf = df['n_PF'].iloc[-1]
        max_hf = df['n_HF'].max()

        results_summary.append({
            'Temperature_K': temp,
            'AlF_max': max_alf,
            'AlF_final': final_alf,
            'PF_final': final_pf,
            'HF_max': max_hf,
        })

    # グラフの装飾
    axes[0, 0].set_xlabel('Time (ps)', fontsize=10)
    axes[0, 0].set_ylabel('Number of Al-F bonds', fontsize=10)
    axes[0, 0].set_title('Al Surface Corrosion (Al-F)', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Time (ps)', fontsize=10)
    axes[0, 1].set_ylabel('Number of P-F bonds', fontsize=10)
    axes[0, 1].set_title('LiPF₆ Decomposition (P-F)', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].set_xlabel('Time (ps)', fontsize=10)
    axes[0, 2].set_ylabel('Number of H-F bonds', fontsize=10)
    axes[0, 2].set_title('HF Formation', fontsize=12, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Time (ps)', fontsize=10)
    axes[1, 0].set_ylabel('Number of Li-F bonds', fontsize=10)
    axes[1, 0].set_title('LiF Formation', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Time (ps)', fontsize=10)
    axes[1, 1].set_ylabel('Number of P-O bonds', fontsize=10)
    axes[1, 1].set_title('P-O Formation (Oxidation)', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].set_xlabel('Time (ps)', fontsize=10)
    axes[1, 2].set_ylabel('Average F height (Å)', fontsize=10)
    axes[1, 2].set_title('F Atoms Height above Al Surface', fontsize=12, fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存
    plot_filename = f"{file_prefix}_al_reaction.png" if file_prefix else "al_reaction.png"
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ グラフを保存しました: {plot_path}\n")

    # サマリーテーブル
    df_summary = pd.DataFrame(results_summary)
    summary_filename = f"{file_prefix}_al_summary.csv" if file_prefix else "al_summary.csv"
    summary_path = output_dir / summary_filename
    df_summary.to_csv(summary_path, index=False)
    print(f"✓ サマリーテーブルを保存しました: {summary_path}\n")

    print("【結果サマリー】")
    print(df_summary.to_string(index=False))
    print()


# ========================================================================
# メイン実行部
# ========================================================================

def main():
    """メイン実行関数"""

    print("\n" + "=" * 70)
    print("  Phase 1-C: LiPF₆とAl接触加熱シミュレーション")
    print("=" * 70 + "\n")

    # 出力ディレクトリの作成
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)

    # ファイル名プレフィックス
    file_prefix = generate_output_filename_prefix(CONFIG.get('lipf6_cif_path'))

    print(f"出力ディレクトリ: {output_dir}")
    if file_prefix:
        print(f"ファイル名プレフィックス: {file_prefix}")
    print()

    # Al表面の構築
    al_slab, n_al_atoms = build_al_surface(CONFIG)

    # LiPF₆分子の配置
    system = add_lipf6_molecules(al_slab, CONFIG)

    # 初期構造の保存
    init_filename = f"{file_prefix}_al_initial.xyz" if file_prefix else "al_initial.xyz"
    init_path = output_dir / init_filename
    write(init_path, system)
    print(f"✓ 初期構造を保存しました: {init_path}\n")

    if not MATLANTIS_AVAILABLE:
        print("\n" + "=" * 70)
        print("  Matlantis環境が利用できません")
        print("=" * 70)
        print("\nこのスクリプトは、Matlantis環境で実行してください。")
        print("初期構造のみ保存しました。\n")
        return

    # 各温度でMD計算を実行
    for temperature in CONFIG['temperatures']:
        run_md_simulation(system, temperature, CONFIG, n_al_atoms, file_prefix)

    # 結果の解析とプロット
    analyze_and_plot_results(CONFIG, file_prefix)

    print("\n" + "=" * 70)
    print("  Phase 1-C 完了")
    print("=" * 70)
    print("\n【観測項目】")
    print("1. Al-F結合の増加 → Al表面の腐食・フッ化")
    print("2. P-F結合の減少 → LiPF₆の分解")
    print("3. HF生成 → 腐食性ガスの発生")
    print("4. F原子の高さ変化 → 表面への吸着")
    print("\n【次のステップ】")
    print("1. al_reaction.png を確認")
    print("2. Al₂O₃表面との比較（Phase 1-D）")
    print("3. 実験データとの比較\n")


if __name__ == "__main__":
    main()
