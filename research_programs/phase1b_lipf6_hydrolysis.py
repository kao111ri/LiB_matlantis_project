"""
Phase 1-B: LiPF₆加水分解検証プログラム

目的:
  - LiPF₆結晶とH₂Oの混合系を加熱し、HF生成反応を追跡
  - Matlantisポテンシャルの妥当性を検証
  - 350K(温水相当)と800K(高温)での反応速度を比較

確認事項:
  1. 350K でのHF生成速度
  2. 副生成物(LiF, POF₃)の確認
  3. 同条件でLiPF₆なし(PVDFのみ)との比較
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
from ase.build import molecule
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
    # LiPF₆結晶ファイル (環境に合わせて変更してください)
    'lipf6_cif_path': "LiPF6.cif",  # またはLiPF₆の構造ファイル

    # シミュレーション条件
    'n_lipf6_molecules': 5,     # LiPF₆分子数
    'n_water_molecules': 10,    # H₂O分子数
    'cell_size': 20.0,          # セルサイズ (Å)

    # MD計算パラメータ
    'temperatures': [350.0, 800.0],  # 計算する温度 (K)
    'timestep': 0.5,            # タイムステップ (fs)
    'simulation_time': 20.0,    # シミュレーション時間 (ps)
    'traj_freq': 100,           # Trajectory保存頻度
    'logger_interval': 100,     # ログ出力頻度

    # Matlantis設定
    'model_version': 'v7.0.0',
    'calc_mode': 'CRYSTAL_U0',

    # 出力ディレクトリ
    'output_dir': 'phase1b_lipf6_hydrolysis_results',
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


class TrackReaction(MDExtensionBase):
    """反応生成物（HF, LiF, P-O結合）を追跡するクラス"""

    def __init__(self, fname: str, dirout: str = '.', stdout: bool = True):
        self.fname = fname
        self.dirout = dirout
        self.stdout = stdout
        self.log_path = f'{self.dirout}/{self.fname}_reaction.log'

    def __call__(self, system, integrator):
        n_step = system.current_total_step
        sim_time_ps = system.current_total_time / 1000.0
        atoms = system.ase_atoms

        if n_step == 0:
            hdr = 'step,time[ps],n_HF,n_LiF,n_PO,n_H2O'
            with open(self.log_path, 'w') as f:
                f.write(f'{hdr}\n')

        # 原子インデックスの取得
        h_idx = [a.index for a in atoms if a.symbol == 'H']
        f_idx = [a.index for a in atoms if a.symbol == 'F']
        li_idx = [a.index for a in atoms if a.symbol == 'Li']
        p_idx = [a.index for a in atoms if a.symbol == 'P']
        o_idx = [a.index for a in atoms if a.symbol == 'O']

        n_hf = n_lif = n_po = n_h2o = 0

        try:
            # 全原子間距離行列の計算
            all_dists = atoms.get_all_distances(mic=True)

            # HF生成（< 1.0Å）
            if h_idx and f_idx:
                hf_dists = all_dists[np.ix_(h_idx, f_idx)]
                n_hf = (hf_dists < 1.0).sum()

            # LiF生成（< 1.8Å）
            if li_idx and f_idx:
                lif_dists = all_dists[np.ix_(li_idx, f_idx)]
                n_lif = (lif_dists < 1.8).sum()

            # P-O結合（POF3等）（< 1.6Å）
            if p_idx and o_idx:
                po_dists = all_dists[np.ix_(p_idx, o_idx)]
                n_po = (po_dists < 1.6).sum()

            # H2O分子数の推定（O-H結合が2つ）
            if o_idx and h_idx:
                oh_dists = all_dists[np.ix_(o_idx, h_idx)]
                oh_bonds = (oh_dists < 1.1)
                h_per_o = oh_bonds.sum(axis=1)
                n_h2o = (h_per_o == 2).sum()

        except Exception as e:
            print(f"REACTION LOG ERROR: {e}")

        # 書き込み
        line = f'{n_step:8d},{sim_time_ps:7.2f},{n_hf:5d},{n_lif:5d},{n_po:5d},{n_h2o:5d}'
        with open(self.log_path, 'a') as f:
            f.write(f'{line}\n')

        if self.stdout and n_step % 500 == 0:  # 500ステップごとに出力
            print(f"REACTION: t={sim_time_ps:.2f}ps HF={n_hf} LiF={n_lif} PO={n_po} H2O={n_h2o}")


# ========================================================================
# システム構築関数
# ========================================================================

def build_lipf6_water_system(config: Dict) -> Atoms:
    """
    LiPF₆とH₂Oの混合系を構築する

    Args:
        config: 設定パラメータの辞書

    Returns:
        構築された系のAtomsオブジェクト
    """
    print("\n=== システム構築 ===\n")

    # LiPF₆結晶の読み込み (もし利用可能なら)
    cif_path = config['lipf6_cif_path']
    if os.path.exists(cif_path):
        print(f"✓ LiPF₆結晶ファイルを読み込みます: {cif_path}")
        lipf6_crystal = read(cif_path)
        system = lipf6_crystal.copy()
    else:
        print(f"✗ LiPF₆結晶ファイルが見つかりません: {cif_path}")
        print("  代わりに、分子から構築します（簡易版）\n")

        # 簡易的なLiPF₆分子の構築 (実際にはCIFファイルを使用することを推奨)
        # ここでは仮の構造を作成
        system = Atoms()

        # LiとPとFを手動配置する簡易版
        # 注: 実際にはCIFファイルまたは分子構造データを使用してください
        n_lipf6 = config['n_lipf6_molecules']
        cell_size = config['cell_size']

        for i in range(n_lipf6):
            # 簡易的なLiPF₆配置（実際の構造とは異なります）
            center = np.random.rand(3) * cell_size * 0.8 + cell_size * 0.1
            system.append(Atoms('Li', positions=[center]))

            # Pを配置
            p_pos = center + np.array([1.5, 0, 0])
            system.append(Atoms('P', positions=[p_pos]))

            # 6個のFを配置（八面体配置の簡易版）
            f_offsets = np.array([
                [0.8, 0, 0], [-0.8, 0, 0],
                [0, 0.8, 0], [0, -0.8, 0],
                [0, 0, 0.8], [0, 0, -0.8]
            ])
            for offset in f_offsets:
                f_pos = p_pos + offset
                system.append(Atoms('F', positions=[f_pos]))

        print(f"✓ {n_lipf6} 個のLiPF₆分子を配置しました（簡易構造）")

    # セルの設定
    cell_size = config['cell_size']
    system.set_cell([cell_size, cell_size, cell_size], scale_atoms=False)
    system.center()
    system.set_pbc(True)

    # H₂O分子の追加
    h2o_template = molecule('H2O')
    n_water = config['n_water_molecules']

    print(f"✓ {n_water} 個のH₂O分子を追加します...")

    for i in range(n_water):
        h2o = h2o_template.copy()
        pos = np.random.rand(3) * cell_size
        h2o.translate(pos)
        # ランダムな回転
        h2o.rotate(np.random.rand() * 360, 'x')
        h2o.rotate(np.random.rand() * 360, 'y')
        system.extend(h2o)

    print(f"\n✓ システム構築完了")
    print(f"  組成: {system.get_chemical_formula()}")
    print(f"  原子数: {len(system)}")
    print(f"  セルサイズ: {cell_size:.2f} Å\n")

    return system


# ========================================================================
# MD実行関数
# ========================================================================

def run_md_simulation(atoms: Atoms, temperature: float, config: Dict, file_prefix: str = "") -> str:
    """
    NVT-MD シミュレーションを実行する

    Args:
        atoms: 初期構造
        temperature: 温度 (K)
        config: 設定パラメータ
        file_prefix: ファイル名プレフィックス（入力ファイル名由来）

    Returns:
        出力ファイル名のプレフィックス
    """
    if not MATLANTIS_AVAILABLE:
        print("✗ Matlantis環境が利用できないため、シミュレーションをスキップします")
        return ""

    print(f"\n=== MD計算開始: {temperature} K ===\n")

    # ファイル名（プレフィックスを追加）
    fname_base = f"{file_prefix}_" if file_prefix else ""
    fname = f"{fname_base}lipf6_h2o_{int(temperature)}K"
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
    logger_reaction = TrackReaction(fname, dirout=output_dir, stdout=True)

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
        file_prefix: ファイル名プレフィックス（入力ファイル名由来）
    """
    print("\n=== 結果解析・グラフ作成 ===\n")

    output_dir = Path(config['output_dir'])
    temperatures = config['temperatures']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    results_summary = []

    for temp in temperatures:
        fname_base = f"{file_prefix}_" if file_prefix else ""
        fname = f"{fname_base}lipf6_h2o_{int(temp)}K"
        reaction_log = output_dir / f"{fname}_reaction.log"

        if not reaction_log.exists():
            print(f"✗ ログファイルが見つかりません: {reaction_log}")
            continue

        # データ読み込み
        df = pd.read_csv(reaction_log)
        label = f"{int(temp)} K"

        # HF生成
        axes[0, 0].plot(df['time[ps]'], df['n_HF'], 'o-', label=label, markersize=3)

        # LiF生成
        axes[0, 1].plot(df['time[ps]'], df['n_LiF'], 'o-', label=label, markersize=3)

        # P-O結合
        axes[1, 0].plot(df['time[ps]'], df['n_PO'], 'o-', label=label, markersize=3)

        # H2O残存
        axes[1, 1].plot(df['time[ps]'], df['n_H2O'], 'o-', label=label, markersize=3)

        # サマリー
        max_hf = df['n_HF'].max()
        final_hf = df['n_HF'].iloc[-1]
        hf_first_time = df[df['n_HF'] > 0]['time[ps]'].min() if (df['n_HF'] > 0).any() else np.nan

        results_summary.append({
            'Temperature_K': temp,
            'HF_first_detection_ps': hf_first_time,
            'HF_max': max_hf,
            'HF_final': final_hf,
        })

    # グラフの装飾
    axes[0, 0].set_xlabel('Time (ps)', fontsize=12)
    axes[0, 0].set_ylabel('Number of HF bonds', fontsize=12)
    axes[0, 0].set_title('HF Generation', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Time (ps)', fontsize=12)
    axes[0, 1].set_ylabel('Number of LiF bonds', fontsize=12)
    axes[0, 1].set_title('LiF Formation', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Time (ps)', fontsize=12)
    axes[1, 0].set_ylabel('Number of P-O bonds', fontsize=12)
    axes[1, 0].set_title('POF₃ Formation (P-O)', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Time (ps)', fontsize=12)
    axes[1, 1].set_ylabel('Number of H₂O molecules', fontsize=12)
    axes[1, 1].set_title('H₂O Consumption', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # ファイル名にプレフィックスを追加
    plot_filename = f"{file_prefix}_reaction_comparison.png" if file_prefix else "reaction_comparison.png"
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ グラフを保存しました: {plot_path}\n")

    # サマリーテーブル
    df_summary = pd.DataFrame(results_summary)
    summary_filename = f"{file_prefix}_reaction_summary.csv" if file_prefix else "reaction_summary.csv"
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

    print("\n" + "=" * 60)
    print("  Phase 1-B: LiPF₆加水分解検証")
    print("=" * 60 + "\n")

    # 出力ディレクトリの作成（共通）
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)

    # 入力ファイル名に基づいてファイル名プレフィックスを生成
    file_prefix = generate_output_filename_prefix(CONFIG.get('lipf6_cif_path'))

    print(f"出力ディレクトリ: {output_dir}")
    if file_prefix:
        print(f"ファイル名プレフィックス: {file_prefix}")
    print()

    # システム構築
    system = build_lipf6_water_system(CONFIG)

    # 初期構造の保存（ファイル名にプレフィックスを追加）
    init_filename = f"{file_prefix}_initial_structure.xyz" if file_prefix else "initial_structure.xyz"
    init_path = output_dir / init_filename
    write(init_path, system)
    print(f"✓ 初期構造を保存しました: {init_path}\n")

    if not MATLANTIS_AVAILABLE:
        print("\n" + "=" * 60)
        print("  Matlantis環境が利用できません")
        print("=" * 60)
        print("\nこのスクリプトは、Matlantis環境で実行してください。")
        print("初期構造のみ保存しました。\n")
        return

    # 各温度でMD計算を実行
    for temperature in CONFIG['temperatures']:
        run_md_simulation(system, temperature, CONFIG, file_prefix)

    # 結果の解析とプロット
    analyze_and_plot_results(CONFIG, file_prefix)

    print("\n" + "=" * 60)
    print("  Phase 1-B 完了")
    print("=" * 60)
    print("\n【確認事項】")
    print("1. 350K でHF生成が観測されるか")
    print("2. 800K でのHF生成速度が350Kより速いか")
    print("3. 副生成物(LiF, POF₃)が生成されているか")
    print("\n【次のステップ】")
    print("1. reaction_comparison.png を確認")
    print("2. 同条件でPVDFのみの系と比較")
    print("3. 実験データと比較\n")


if __name__ == "__main__":
    main()
