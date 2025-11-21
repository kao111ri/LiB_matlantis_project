"""
Phase 1-C: LiPF₆とAlの接触加熱シミュレーション（Packmol版）

目的:
  - LiPF₆とAl金属表面の接触反応を調査
  - 加熱条件下での反応生成物の追跡
  - Al表面の酸化・腐食の観測
  - 複数温度条件での反応速度の比較

★修正版: Packmolを使用した水分子の充填機能を追加
★オプションで水分子を追加可能（LiPF6の加水分解反応を観測）

確認事項:
  1. LiPF₆の分解生成物（HF, PF₃, PF₅など）
  2. Al表面の反応（AlF₃生成など）
  3. 温度依存性（300K, 400K, 600K, 800K）
  4. 水分子存在下での加水分解反応
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple, Optional

from ase import Atoms
from ase.build import bulk, surface
from ase.io import read, write
from ase import units
from ase.constraints import FixAtoms

# プロジェクトのutilsをインポート
sys.path.append(str(Path(__file__).parent.parent / "LiB2_structure_ipynb"))
try:
    from utils.io_utils import generate_output_filename_prefix
except ImportError:
    def generate_output_filename_prefix(path):
        if path:
            return Path(path).stem
        return ""

# Packmol関連のインポート（水充填用）
try:
    from utils.packmol_utils import (
        fill_box_with_packmol,
        check_packmol_command
    )
    PACKMOL_AVAILABLE = check_packmol_command()
    if PACKMOL_AVAILABLE:
        print("✓ Packmolが利用可能です（水充填機能が使用できます）")
except ImportError:
    PACKMOL_AVAILABLE = False
    print("注意: packmol_utils が見つかりません（水充填機能は使用できません）")

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
    # ★ 構造ファイルのパス
    'substrate_type': 'Al',  # 'Al' または 'Al2O3' を選択
    'al_structure_path': "/home/jovyan/Kaori/MD/LiB_2/structure/Al_cell.cif",
    'al2o3_structure_path': "/home/jovyan/Kaori/MD/LiB_2/structure/Al2O3_cell.cif",
    'lipf6_cif_path': "/home/jovyan/Kaori/MD/input/LiPF6.cif",

    # 基板のリピート設定
    'substrate_repeat': (2, 2, 1),  # (nx, ny, nz) のスーパーセル化

    # Al基板設定（ase.buildで作成する場合のフォールバック）
    'al_surface_size': (3, 3, 3),
    'al_vacuum': 10.0,

    # LiPF₆配置設定
    'n_lipf6_molecules': 3,
    'lipf6_height_above_surface': 3.0,
    'lipf6_use_full_unit_cell': True,

    # ★ 水充填設定（Packmol）
    'add_water': False,  # True にすると水分子を追加
    'water_density_g_cm3': 0.8,  # 水の密度 (g/cm³)
    'water_tolerance': 2.0,  # Packmolの原子間最小距離 (Å)
    'water_seed': 98765,  # 乱数シード
    'structure_cell_size': None,  # 構造全体のセルサイズ（Noneで自動）
    'water_fill_cell_size': None,  # 水充填領域サイズ（Noneで全体）
    # 使用例:
    # 'structure_cell_size': [30.0, 30.0, 40.0],  # セル全体
    # 'water_fill_cell_size': [25.0, 25.0, 35.0],  # 水充填領域

    # MD計算パラメータ
    'temperatures': [300.0, 400.0, 600.0, 800.0],
    'timestep': 0.5,
    'simulation_time': 30.0,
    'traj_freq': 100,
    'logger_interval': 100,

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
            all_dists = atoms.get_all_distances(mic=True)

            if al_idx and f_idx:
                alf_dists = all_dists[np.ix_(al_idx, f_idx)]
                n_alf = (alf_dists < 2.0).sum()

            if p_idx and f_idx:
                pf_dists = all_dists[np.ix_(p_idx, f_idx)]
                n_pf = (pf_dists < 1.8).sum()

            if h_idx and f_idx:
                hf_dists = all_dists[np.ix_(h_idx, f_idx)]
                n_hf = (hf_dists < 1.0).sum()

            if li_idx and f_idx:
                lif_dists = all_dists[np.ix_(li_idx, f_idx)]
                n_lif = (lif_dists < 1.8).sum()

            if p_idx and o_idx:
                po_dists = all_dists[np.ix_(p_idx, o_idx)]
                n_po = (po_dists < 1.6).sum()

            if f_idx and al_idx:
                al_positions = atoms.positions[al_idx]
                f_positions = atoms.positions[f_idx]
                al_avg_z = al_positions[:, 2].mean()
                f_avg_z = f_positions[:, 2].mean()
                avg_f_height = f_avg_z - al_avg_z

        except Exception as e:
            print(f"REACTION LOG ERROR: {e}")

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

def build_substrate_surface(config: Dict) -> Tuple[Atoms, int]:
    """基板表面を構築する"""
    print("\n=== 基板表面の構築 ===\n")

    substrate_type = config['substrate_type']

    if substrate_type == 'Al':
        structure_path = config.get('al_structure_path')
    elif substrate_type == 'Al2O3':
        structure_path = config.get('al2o3_structure_path')
    else:
        raise ValueError(f"未対応の基板タイプ: {substrate_type}")

    if structure_path and os.path.exists(structure_path):
        print(f"✓ 構造ファイルから読み込みます: {structure_path}")
        substrate = read(structure_path)
        print(f"  読み込んだ構造: {substrate.get_chemical_formula()}")
        print(f"  原子数: {len(substrate)}")

        repeat = config.get('substrate_repeat', (1, 1, 1))
        if repeat != (1, 1, 1):
            print(f"  スーパーセル化: {repeat}")
            substrate = substrate.repeat(repeat)
            print(f"  スーパーセル化後の原子数: {len(substrate)}")

        cell_params = substrate.cell.cellpar()
        print(f"  セルサイズ: a={cell_params[0]:.2f}, b={cell_params[1]:.2f}, c={cell_params[2]:.2f} Å")

    else:
        print(f"✗ 構造ファイルが見つかりません: {structure_path}")
        print("  ase.buildで基板を作成します\n")

        if substrate_type != 'Al':
            raise FileNotFoundError(f"{substrate_type}の構造ファイルが必要です")

        size = config['al_surface_size']
        substrate = surface('Al', (1, 0, 0), size[2], vacuum=config['al_vacuum'])
        substrate = substrate.repeat((size[0], size[1], 1))
        print(f"  Al(100)表面を作成: サイズ {size}")
        print(f"  原子数: {len(substrate)}")

    # 最下層の原子を固定
    z_positions = substrate.positions[:, 2]
    min_z = z_positions.min()
    fixed_indices = [i for i, z in enumerate(z_positions) if z < min_z + 3.0]

    if fixed_indices:
        constraint = FixAtoms(indices=fixed_indices)
        substrate.set_constraint(constraint)
        print(f"  固定原子数: {len(fixed_indices)} (z < {min_z + 3.0:.2f} Å)\n")

    return substrate, len(substrate)


def add_lipf6_molecules(substrate: Atoms, config: Dict) -> Atoms:
    """基板表面上にLiPF₆分子を配置する"""
    print("=== LiPF₆分子の配置 ===\n")

    system = substrate.copy()
    height = config['lipf6_height_above_surface']
    use_full_unit_cell = config.get('lipf6_use_full_unit_cell', True)

    substrate_top_z = substrate.positions[:, 2].max()
    cell_params = substrate.cell.cellpar()
    center_x = cell_params[0] / 2
    center_y = cell_params[1] / 2

    cif_path = config['lipf6_cif_path']

    if not os.path.exists(cif_path):
        print(f"✗ LiPF₆結晶ファイルが見つかりません: {cif_path}")
        raise FileNotFoundError(f"LiPF6ファイルが必要です: {cif_path}")

    print(f"✓ LiPF₆結晶ファイルを読み込みます: {cif_path}")
    lipf6_unit = read(cif_path)
    print(f"  読み込んだ構造: {lipf6_unit.get_chemical_formula()}")

    if use_full_unit_cell:
        print(f"  単位格子全体を基板上に配置します")
        lipf6_copy = lipf6_unit.copy()
        com = lipf6_copy.positions.mean(axis=0)
        target_pos = np.array([center_x, center_y, substrate_top_z + height])
        lipf6_copy.translate(target_pos - com)
        system.extend(lipf6_copy)
        print(f"  追加された原子数: {len(lipf6_copy)}")

    print(f"✓ LiPF₆を配置しました")
    print(f"  総原子数: {len(system)}")
    print(f"  組成: {system.get_chemical_formula()}\n")

    return system


def add_water_with_packmol(system: Atoms, config: Dict) -> Atoms:
    """
    Packmolを使用して水分子を充填する

    Args:
        system: 既存の構造（基板+LiPF6）
        config: 設定パラメータ

    Returns:
        水分子が追加された構造
    """
    print("\n=== Packmolによる水分子の充填 ===\n")

    if not PACKMOL_AVAILABLE:
        print("✗ Packmolが利用できません。水の追加をスキップします。")
        return system

    if not config.get('add_water', False):
        print("水の追加が無効になっています（add_water=False）")
        return system

    try:
        solvated_system = fill_box_with_packmol(
            host_atoms=system,
            solvent_type='H2O',
            density_g_cm3=config['water_density_g_cm3'],
            tolerance=config['water_tolerance'],
            seed=config['water_seed'],
            structure_cell_size=config.get('structure_cell_size'),
            water_fill_cell_size=config.get('water_fill_cell_size'),
            verbose=True
        )

        print(f"\n✓ 水分子の充填が完了しました")
        print(f"  最終組成: {solvated_system.get_chemical_formula()}")
        print(f"  総原子数: {len(solvated_system)}")

        # 追加された水分子数を計算
        n_water_atoms = len(solvated_system) - len(system)
        n_water_molecules = n_water_atoms // 3
        print(f"  追加された水分子数: {n_water_molecules} ({n_water_atoms} 原子)\n")

        return solvated_system

    except Exception as e:
        print(f"✗ 水の充填中にエラーが発生しました: {e}")
        print("  元の構造を使用します\n")
        return system


# ========================================================================
# MD実行関数
# ========================================================================

def run_md_simulation(atoms: Atoms, temperature: float, config: Dict,
                      n_substrate_atoms: int, file_prefix: str = "") -> str:
    """NVT-MD シミュレーションを実行する"""
    if not MATLANTIS_AVAILABLE:
        print("✗ Matlantis環境が利用できないため、シミュレーションをスキップします")
        return ""

    print(f"\n=== MD計算開始: {temperature} K ===\n")

    substrate_name = config['substrate_type']
    fname_base = f"{file_prefix}_" if file_prefix else ""
    water_suffix = "_water" if config.get('add_water', False) else ""
    fname = f"{fname_base}lipf6_{substrate_name}_{int(temperature)}K{water_suffix}"
    output_dir = config['output_dir']

    n_steps = int(config['simulation_time'] * 1000 / config['timestep'])

    estimator_fn = pfp_estimator_fn(
        model_version=config['model_version'],
        calc_mode=config['calc_mode']
    )

    system = ASEMDSystem(atoms.copy(), step=0, time=0.0)

    integrator = NVTBerendsenIntegrator(
        timestep=config['timestep'],
        temperature=temperature,
        taut=100.0,
        fixcm=True,
    )

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

    logger_std = PrintWriteLog(fname, dirout=output_dir, stdout=False)
    logger_reaction = TrackAlReaction(fname, dirout=output_dir, stdout=True,
                                     n_al_atoms=n_substrate_atoms)

    print(f"ステップ数: {n_steps} ({config['simulation_time']} ps)")
    print(f"温度: {temperature} K\n")

    md(system, extensions=[(logger_std, config['logger_interval']),
                           (logger_reaction, config['logger_interval'])])

    print(f"\n✓ MD計算完了: {fname}\n")

    return fname


# ========================================================================
# 結果解析・プロット関数
# ========================================================================

def analyze_and_plot_results(config: Dict, file_prefix: str = ""):
    """シミュレーション結果を解析してプロットする"""
    print("\n=== 結果解析・グラフ作成 ===\n")

    output_dir = Path(config['output_dir'])
    temperatures = config['temperatures']
    substrate_name = config['substrate_type']
    water_suffix = "_water" if config.get('add_water', False) else ""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    title_suffix = " (with H₂O)" if config.get('add_water', False) else ""
    fig.suptitle(f'LiPF₆ + {substrate_name} Surface Reaction Analysis{title_suffix}',
                 fontsize=16, fontweight='bold')

    results_summary = []

    for temp in temperatures:
        fname_base = f"{file_prefix}_" if file_prefix else ""
        fname = f"{fname_base}lipf6_{substrate_name}_{int(temp)}K{water_suffix}"
        reaction_log = output_dir / f"{fname}_reaction.log"

        if not reaction_log.exists():
            print(f"✗ ログファイルが見つかりません: {reaction_log}")
            continue

        df = pd.read_csv(reaction_log)
        label = f"{int(temp)} K"

        axes[0, 0].plot(df['time[ps]'], df['n_AlF'], 'o-', label=label, markersize=3)
        axes[0, 1].plot(df['time[ps]'], df['n_PF'], 'o-', label=label, markersize=3)
        axes[0, 2].plot(df['time[ps]'], df['n_HF'], 'o-', label=label, markersize=3)
        axes[1, 0].plot(df['time[ps]'], df['n_LiF'], 'o-', label=label, markersize=3)
        axes[1, 1].plot(df['time[ps]'], df['n_PO'], 'o-', label=label, markersize=3)
        axes[1, 2].plot(df['time[ps]'], df['avg_F_height[A]'], 'o-', label=label, markersize=3)

        results_summary.append({
            'Temperature_K': temp,
            'AlF_max': df['n_AlF'].max(),
            'AlF_final': df['n_AlF'].iloc[-1],
            'PF_final': df['n_PF'].iloc[-1],
            'HF_max': df['n_HF'].max(),
        })

    # グラフの装飾
    axes[0, 0].set_xlabel('Time (ps)', fontsize=10)
    axes[0, 0].set_ylabel(f'Number of {substrate_name}-F bonds', fontsize=10)
    axes[0, 0].set_title(f'{substrate_name} Surface Corrosion', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Time (ps)', fontsize=10)
    axes[0, 1].set_ylabel('Number of P-F bonds', fontsize=10)
    axes[0, 1].set_title('LiPF₆ Decomposition', fontsize=12, fontweight='bold')
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
    axes[1, 1].set_title('P-O Formation', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].set_xlabel('Time (ps)', fontsize=10)
    axes[1, 2].set_ylabel('Average F height (Å)', fontsize=10)
    axes[1, 2].set_title(f'F Height above {substrate_name}', fontsize=12, fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_filename = f"{file_prefix}_{substrate_name}_reaction{water_suffix}.png" if file_prefix else f"{substrate_name}_reaction{water_suffix}.png"
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ グラフを保存しました: {plot_path}\n")

    df_summary = pd.DataFrame(results_summary)
    summary_filename = f"{file_prefix}_{substrate_name}_summary{water_suffix}.csv" if file_prefix else f"{substrate_name}_summary{water_suffix}.csv"
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
    print("  Phase 1-C: LiPF₆と基板接触加熱シミュレーション（Packmol版）")
    if CONFIG.get('add_water', False):
        print("  ★ 水分子を追加してLiPF₆の加水分解反応を観測")
    print("=" * 70 + "\n")

    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)

    file_prefix = generate_output_filename_prefix(CONFIG.get('lipf6_cif_path'))

    print(f"基板タイプ: {CONFIG['substrate_type']}")
    print(f"水の追加: {'有効' if CONFIG.get('add_water', False) else '無効'}")
    if CONFIG.get('add_water', False):
        print(f"  水の密度: {CONFIG['water_density_g_cm3']} g/cm³")
    print(f"出力ディレクトリ: {output_dir}")
    print()

    # 基板表面の構築
    substrate, n_substrate_atoms = build_substrate_surface(CONFIG)

    # LiPF₆分子の配置
    system = add_lipf6_molecules(substrate, CONFIG)

    # ★ 水分子の追加（オプション）
    system = add_water_with_packmol(system, CONFIG)

    # 初期構造の保存
    substrate_name = CONFIG['substrate_type']
    water_suffix = "_water" if CONFIG.get('add_water', False) else ""
    init_filename = f"{file_prefix}_{substrate_name}_initial{water_suffix}.xyz" if file_prefix else f"{substrate_name}_initial{water_suffix}.xyz"
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
        run_md_simulation(system, temperature, CONFIG, n_substrate_atoms, file_prefix)

    # 結果の解析とプロット
    analyze_and_plot_results(CONFIG, file_prefix)

    print("\n" + "=" * 70)
    print("  Phase 1-C 完了")
    print("=" * 70)
    print("\n【観測項目】")
    print(f"1. {substrate_name}-F結合の増加 → {substrate_name}表面の腐食・フッ化")
    print("2. P-F結合の減少 → LiPF₆の分解")
    print("3. HF生成 → 腐食性ガスの発生")
    if CONFIG.get('add_water', False):
        print("4. LiPF₆の加水分解反応（水存在下）")
    print("\n【次のステップ】")
    print(f"1. グラフを確認: {substrate_name}_reaction{water_suffix}.png")
    print("2. 水あり/なしでの比較")
    print("3. 異なる基板との比較\n")

    print("【使用方法】")
    print("水を追加する場合:")
    print("  CONFIG['add_water'] = True")
    print("  CONFIG['water_density_g_cm3'] = 0.8  # 密度を調整")
    print("  CONFIG['structure_cell_size'] = [30.0, 30.0, 40.0]  # セルサイズ指定（オプション）")
    print("  CONFIG['water_fill_cell_size'] = [25.0, 25.0, 35.0]  # 水充填領域指定（オプション）\n")


if __name__ == "__main__":
    main()
