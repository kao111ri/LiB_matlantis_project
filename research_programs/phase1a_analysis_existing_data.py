"""
Phase 1-A 改良版: 過剰加速判定機能を追加

新規追加機能:
  1. 複数ファイル対応（Al-PVDF、Al-23PVDF、H2O入りなど）
  2. Al2O3以外にも対応（金属Al、H2O含有系など）
  3. 分子種同定（CF₄, C₂F₄, ベンゼン環、CO、CO2、H2など高温特有の生成物検出）
  4. 反応速度定数の計算（活性化エネルギー推定用）
  5. 文献値との自動比較
  6. 過剰加速判定レポート生成

確認事項:
  1. HF生成開始時刻
  2. AlF₃生成位置(表面 vs 界面)
  3. F原子のAl層貫通可否
  4. 高温特有の生成物の検出
  5. 時間スケール換算の妥当性
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from ase.io import read
from ase.geometry import get_distances
import warnings
warnings.filterwarnings('ignore')

# プロジェクトのutilsをインポート
sys.path.append(str(Path(__file__).parent.parent / "LiB2_structure_ipynb"))
from utils.io_utils import generate_output_filename_prefix

# ========================================================================
# 設定パラメータ
# ========================================================================

# 入力ファイルのパス（複数ファイル対応）
# 環境に合わせて変更してください
INPUT_TRAJ_PATHS = [
    "Al2O3-PVDF_md_1600K_fix_al.traj",
    # "Al-PVDF_md_1600K.traj",
    # "Al-23PVDF_md_H2O_1600K_fix_al.traj",
    # 必要に応じて追加
]

# 出力ディレクトリ
OUTPUT_DIR = "phase1a_analysis_results"

# 解析パラメータ
ANALYSIS_PARAMS = {
    'HF_cutoff': 1.0,        # HF結合の判定距離 (Å)
    'AlF_cutoff': 2.0,       # Al-F結合の判定距離 (Å)
    'OH_cutoff': 1.1,        # O-H結合の判定距離 (Å)
    'CF_cutoff': 1.6,        # C-F結合判定 (Å)
    'CC_cutoff': 1.8,        # C-C結合判定 (Å)
    'CO_cutoff': 1.5,        # C-O結合判定 (Å)
    'HH_cutoff': 1.0,        # H-H結合判定 (Å)
    'frame_interval': 10,    # 解析するフレーム間隔(全フレーム解析は重いため)
}

# 文献値データベース（600-800K実験）
LITERATURE_VALUES = {
    'PVDF_decomposition_temp': (589, 723),  # K, 316-450°C
    'expected_products': {
        'HF': 'major',        # 主生成物
        'CF4': 'trace',       # 微量（<5%）
        'C2F4': 'minor',      # 少量（5-15%）
        'aromatic': 'none',   # なし
        'char': 'minor',      # 炭素残渣
        'CO': 'trace',        # 微量（酸素がある場合）
        'CO2': 'trace',       # 微量（酸素がある場合）
        'H2': 'trace',        # 微量
    },
    'Ea_deHF_eV': (1.8, 2.5),  # 脱HF反応の活性化エネルギー範囲
}

# ========================================================================
# 解析関数
# ========================================================================

def identify_molecules(atoms, params: Dict) -> Dict[str, float]:
    """
    原子系から分子種を同定

    Args:
        atoms: ASE Atoms オブジェクト
        params: 解析パラメータ

    Returns:
        分子種のカウント辞書
    """
    molecules = {
        'HF': 0,
        'CF4': 0,
        'C2F4': 0,
        'C3F6': 0,
        'benzene_ring': 0,  # C6環
        'AlF3': 0,
        'Al_metal_reduced': 0,  # 酸化物から還元されたAl
        'large_C_cluster': 0,  # 炭素クラスター(>10原子)
        'CO': 0,
        'CO2': 0,
        'H2': 0,
        'H2O': 0,
    }

    positions = atoms.positions
    symbols = atoms.symbols

    # 原子インデックス取得
    H_idx = np.where(symbols == 'H')[0]
    F_idx = np.where(symbols == 'F')[0]
    C_idx = np.where(symbols == 'C')[0]
    Al_idx = np.where(symbols == 'Al')[0]
    O_idx = np.where(symbols == 'O')[0]

    # === HF検出 ===
    if len(H_idx) > 0 and len(F_idx) > 0:
        h_pos = positions[H_idx]
        f_pos = positions[F_idx]
        dists = get_distances(h_pos, f_pos, cell=atoms.cell, pbc=atoms.pbc)[1]
        molecules['HF'] = (dists < params['HF_cutoff']).sum()

    # === CF₄検出 ===
    for c_idx in C_idx:
        c_pos = positions[c_idx]
        if len(F_idx) > 0:
            dists = np.linalg.norm(positions[F_idx] - c_pos, axis=1)
            n_F_bonded = (dists < params['CF_cutoff']).sum()

            if n_F_bonded == 4:
                molecules['CF4'] += 1

    # === C₂F₄検出（2つのC原子がそれぞれ2つのF原子と結合）===
    if len(C_idx) >= 2:
        c_dists = get_distances(positions[C_idx], positions[C_idx], cell=atoms.cell, pbc=atoms.pbc)[1]
        np.fill_diagonal(c_dists, np.inf)

        for i, c1_idx in enumerate(C_idx):
            # C-C結合を持つ隣接C原子
            c2_candidates = C_idx[c_dists[i] < params['CC_cutoff']]

            for c2_idx in c2_candidates:
                c1_pos = positions[c1_idx]
                c2_pos = positions[c2_idx]

                if len(F_idx) > 0:
                    f_dists_c1 = np.linalg.norm(positions[F_idx] - c1_pos, axis=1)
                    f_dists_c2 = np.linalg.norm(positions[F_idx] - c2_pos, axis=1)

                    n_F_c1 = (f_dists_c1 < params['CF_cutoff']).sum()
                    n_F_c2 = (f_dists_c2 < params['CF_cutoff']).sum()

                    if n_F_c1 == 2 and n_F_c2 == 2:
                        molecules['C2F4'] += 0.5  # ダブルカウント防止

    # === CO検出 ===
    if len(C_idx) > 0 and len(O_idx) > 0:
        for c_idx in C_idx:
            c_pos = positions[c_idx]
            dists = np.linalg.norm(positions[O_idx] - c_pos, axis=1)
            n_O = (dists < params['CO_cutoff']).sum()

            if n_O == 1:
                molecules['CO'] += 1
            elif n_O == 2:
                molecules['CO2'] += 0.5  # COが2つカウントされないように

    # === CO₂検出 ===
    if len(C_idx) > 0 and len(O_idx) >= 2:
        for c_idx in C_idx:
            c_pos = positions[c_idx]
            dists = np.linalg.norm(positions[O_idx] - c_pos, axis=1)
            n_O = (dists < params['CO_cutoff']).sum()

            if n_O == 2:
                molecules['CO2'] += 1

    # === H₂検出 ===
    if len(H_idx) >= 2:
        h_dists = get_distances(positions[H_idx], positions[H_idx], cell=atoms.cell, pbc=atoms.pbc)[1]
        np.fill_diagonal(h_dists, np.inf)
        molecules['H2'] = ((h_dists < params['HH_cutoff']).sum()) / 2  # ダブルカウント防止

    # === H₂O検出 ===
    if len(H_idx) > 0 and len(O_idx) > 0:
        for o_idx in O_idx:
            o_pos = positions[o_idx]
            dists = np.linalg.norm(positions[H_idx] - o_pos, axis=1)
            n_H = (dists < params['OH_cutoff']).sum()

            if n_H >= 2:
                molecules['H2O'] += 1

    # === ベンゼン環検出（6員環のC原子）===
    if len(C_idx) >= 6:
        try:
            c_dists = get_distances(positions[C_idx], positions[C_idx], cell=atoms.cell, pbc=atoms.pbc)[1]

            # グラフ理論的な環検出（簡易版）
            adjacency = (c_dists < 1.6) & (c_dists > 0.1)
            for i in range(len(C_idx)):
                neighbors = np.where(adjacency[i])[0]
                if len(neighbors) == 2:  # 環構造では各原子は2つの隣接原子を持つ
                    molecules['benzene_ring'] += 1 / 6  # 6原子で1環
        except:
            pass

    # === AlF₃検出 ===
    for al_idx in Al_idx:
        al_pos = positions[al_idx]
        if len(F_idx) > 0:
            dists = np.linalg.norm(positions[F_idx] - al_pos, axis=1)
            n_F = (dists < params['AlF_cutoff']).sum()
            if n_F >= 3:
                molecules['AlF3'] += 1

    # === 金属Al還元検出（酸素配位数の減少）===
    for al_idx in Al_idx:
        al_pos = positions[al_idx]
        if len(O_idx) > 0:
            dists = np.linalg.norm(positions[O_idx] - al_pos, axis=1)
            n_O = (dists < 2.5).sum()  # Al-O配位数

            # 初期のAl₂O₃ではn_O=4-6、還元されると0-2
            if n_O < 2:
                molecules['Al_metal_reduced'] += 1

    # === 大きな炭素クラスター検出 ===
    if len(C_idx) > 10:
        c_dists = get_distances(positions[C_idx], positions[C_idx], cell=atoms.cell, pbc=atoms.pbc)[1]
        adjacency = (c_dists < 2.0) & (c_dists > 0.1)

        # クラスタリング（簡易版）
        visited = set()
        clusters = []

        def dfs(node, cluster):
            visited.add(node)
            cluster.append(node)
            for neighbor in np.where(adjacency[node])[0]:
                if neighbor not in visited:
                    dfs(neighbor, cluster)

        for i in range(len(C_idx)):
            if i not in visited:
                cluster = []
                dfs(i, cluster)
                if len(cluster) > 10:
                    clusters.append(cluster)

        molecules['large_C_cluster'] = len(clusters)

    return molecules


def calculate_reaction_rates(df: pd.DataFrame) -> Dict[str, float]:
    """
    各反応の速度定数を計算

    Args:
        df: 解析結果のDataFrame

    Returns:
        速度定数の辞書 (単位: molecules/ps)
    """
    rates = {}

    # HF生成速度（初期の線形領域でフィット）
    time = df['time_ps'].values
    hf_count = df['n_HF'].values

    # 最初の20%のデータで線形フィット
    n_points = max(3, len(time) // 5)
    if len(time) > n_points:
        try:
            slope, _ = np.polyfit(time[:n_points], hf_count[:n_points], 1)
            rates['HF_formation'] = slope
        except:
            rates['HF_formation'] = 0.0

    # AlF₃生成速度
    if 'n_AlF' in df.columns:
        alf_count = df['n_AlF'].values
        if len(time) > n_points:
            try:
                slope, _ = np.polyfit(time[:n_points], alf_count[:n_points], 1)
                rates['AlF3_formation'] = slope
            except:
                rates['AlF3_formation'] = 0.0

    # F貫通速度
    if 'F_penetration_A' in df.columns:
        penetration = df['F_penetration_A'].values
        if len(time) > n_points and penetration.max() > 0:
            try:
                slope, _ = np.polyfit(time[:n_points], penetration[:n_points], 1)
                rates['F_penetration'] = slope
            except:
                rates['F_penetration'] = 0.0

    return rates


def judge_excessive_acceleration(
    molecules_final: Dict[str, float],
    df: pd.DataFrame,
    T_sim: float = 1600
) -> Dict[str, any]:
    """
    過剰加速の判定を行う

    Args:
        molecules_final: 最終フレームの分子種
        df: 解析結果DataFrame
        T_sim: シミュレーション温度

    Returns:
        判定結果の辞書
    """
    warnings_list = []
    severity = 'OK'  # OK, WARNING, CRITICAL

    lit_values = LITERATURE_VALUES['expected_products']

    # === 判定1: 予期しない生成物 ===
    if molecules_final.get('CF4', 0) > molecules_final.get('HF', 1) * 0.1:
        warnings_list.append(
            f"⚠️ CF₄過剰生成: {molecules_final['CF4']:.0f}個 "
            f"(HFの{molecules_final['CF4']/max(molecules_final.get('HF', 1), 1)*100:.1f}%)"
        )
        severity = 'WARNING'

    if molecules_final.get('benzene_ring', 0) > 0:
        warnings_list.append(
            f"❌ ベンゼン環検出: {molecules_final['benzene_ring']:.1f}個 "
            "(高温特有の環化反応)"
        )
        severity = 'CRITICAL'

    if molecules_final.get('Al_metal_reduced', 0) > 10:
        warnings_list.append(
            f"❌ Al還元: {molecules_final['Al_metal_reduced']:.0f}個の金属Al生成 "
            "(非現実的な還元反応)"
        )
        severity = 'CRITICAL'

    if molecules_final.get('large_C_cluster', 0) > 0:
        warnings_list.append(
            f"⚠️ 大型炭素クラスター: {molecules_final['large_C_cluster']:.0f}個 "
            "(過剰な炭化)"
        )
        if severity == 'OK':
            severity = 'WARNING'

    # === 判定2: 反応速度の異常 ===
    rates = calculate_reaction_rates(df)

    if 'HF_formation' in rates:
        # 1600Kでの速度が異常に速い場合
        if rates['HF_formation'] > 10:  # molecules/ps
            warnings_list.append(
                f"⚠️ HF生成速度が異常に速い: {rates['HF_formation']:.2f} molecules/ps"
            )
            if severity == 'OK':
                severity = 'WARNING'

    # === 判定3: 時間スケール換算 ===
    T_exp = 600  # K
    t_sim = df['time_ps'].max() * 1e-12  # 秒

    # 仮の活性化エネルギー（文献値の中央値）
    Ea_eV = np.mean(LITERATURE_VALUES['Ea_deHF_eV'])
    kB = 8.617e-5  # eV/K

    ratio = np.exp(Ea_eV/kB * (1/T_exp - 1/T_sim))
    t_equivalent = t_sim * ratio

    time_scale_info = {
        'T_sim': T_sim,
        'T_exp': T_exp,
        't_sim_ps': df['time_ps'].max(),
        't_equivalent_hours': t_equivalent / 3600,
        'acceleration_ratio': ratio,
        'Ea_assumed_eV': Ea_eV
    }

    if t_equivalent > 86400:  # 1日以上
        warnings_list.append(
            f"⚠️ 時間スケール換算: {T_sim}K/{df['time_ps'].max():.0f}ps = "
            f"{T_exp}K/{t_equivalent/3600:.1f}時間 (実験的に非現実的)"
        )
        if severity == 'OK':
            severity = 'WARNING'

    return {
        'severity': severity,
        'warnings': warnings_list,
        'time_scale': time_scale_info,
        'molecules_final': molecules_final,
        'rates': rates
    }


def analyze_trajectory(traj_path: str, params: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    拡張版trajectory解析（分子種同定を含む）

    Args:
        traj_path: trajectoryファイルのパス
        params: 解析パラメータの辞書

    Returns:
        解析結果のDataFrameと分子種履歴のタプル
    """
    print(f"=== 拡張Trajectory解析開始: {traj_path} ===\n")

    # trajectoryの読み込み
    try:
        traj = read(traj_path, ":")
        print(f"✓ Trajectoryファイル読み込み完了: {len(traj)} フレーム\n")
    except FileNotFoundError:
        print(f"✗ エラー: ファイルが見つかりません: {traj_path}")
        print("  注: このスクリプトを実際に実行する際は、trajectoryファイルのパスを正しく設定してください。")
        return pd.DataFrame(), []
    except Exception as e:
        print(f"✗ エラー: ファイルの読み込みに失敗しました: {e}")
        return pd.DataFrame(), []

    results = []
    molecules_history = []
    frame_interval = params['frame_interval']

    print(f"解析フレーム間隔: {frame_interval} フレームごと")
    print(f"解析対象フレーム数: {len(traj[::frame_interval])} フレーム\n")
    print("解析中...\n")

    for i, atoms in enumerate(traj[::frame_interval]):
        frame_num = i * frame_interval

        # 原子インデックスの取得
        h_indices = [a.index for a in atoms if a.symbol == 'H']
        f_indices = [a.index for a in atoms if a.symbol == 'F']
        o_indices = [a.index for a in atoms if a.symbol == 'O']
        al_indices = [a.index for a in atoms if a.symbol == 'Al']

        # Al層の検出（Al2O3、Al、またはtagで識別）
        al_substrate_indices = [a.index for a in atoms if a.symbol == 'Al' and hasattr(a, 'tag') and a.tag == 2]

        # tagが設定されていない場合は、Z座標で判定
        if not al_substrate_indices and al_indices:
            al_positions = atoms.positions[al_indices]
            al_z = al_positions[:, 2]
            z_threshold = np.percentile(al_z, 75)  # 上位25%をAl基板と仮定
            al_substrate_indices = [al_indices[j] for j, z in enumerate(al_z) if z > z_threshold]

        # それでもない場合は、全てのAlを基板と見なす
        if not al_substrate_indices:
            al_substrate_indices = al_indices

        # 距離行列の計算
        all_positions = atoms.positions

        # 1. HF生成数
        n_hf = 0
        if h_indices and f_indices:
            h_pos = all_positions[h_indices]
            f_pos = all_positions[f_indices]
            distances_hf = get_distances(h_pos, f_pos, cell=atoms.cell, pbc=atoms.pbc)[1]
            n_hf = (distances_hf < params['HF_cutoff']).sum()

        # 2. Al-F結合(AlF₃)生成数
        n_alf = 0
        if al_substrate_indices and f_indices:
            al_pos = all_positions[al_substrate_indices]
            f_pos = all_positions[f_indices]
            distances_alf = get_distances(al_pos, f_pos, cell=atoms.cell, pbc=atoms.pbc)[1]
            n_alf = (distances_alf < params['AlF_cutoff']).sum()

        # 3. O-H結合数(水生成の指標)
        n_oh = 0
        if o_indices and h_indices:
            o_pos = all_positions[o_indices]
            h_pos = all_positions[h_indices]
            distances_oh = get_distances(o_pos, h_pos, cell=atoms.cell, pbc=atoms.pbc)[1]
            n_oh = (distances_oh < params['OH_cutoff']).sum()

        # 4. F貫通深度
        f_penetration = 0.0
        if f_indices and al_substrate_indices:
            f_z = all_positions[f_indices][:, 2]
            al_surface_z = all_positions[al_substrate_indices][:, 2].max()

            # Al表面より下にあるF原子の深度を計算
            penetrating_f = f_z[f_z < al_surface_z]
            if len(penetrating_f) > 0:
                f_penetration = al_surface_z - penetrating_f.min()

        # 5. 温度(もし利用可能なら)
        try:
            temperature = atoms.get_temperature()
        except:
            temperature = 0.0

        # 6. 新規: 分子種同定
        molecules = identify_molecules(atoms, params)
        molecules_history.append(molecules)

        # 結果を記録
        result_dict = {
            'frame': frame_num,
            'time_ps': frame_num * 0.0001,  # 0.1fs/stepと仮定、適宜調整
            'temperature_K': temperature,
            'n_HF': n_hf,
            'n_AlF': n_alf,
            'n_OH': n_oh,
            'F_penetration_A': f_penetration,
        }

        # 分子種データを追加
        for mol_name, mol_count in molecules.items():
            result_dict[f'mol_{mol_name}'] = mol_count

        results.append(result_dict)

        # 進捗表示
        if (i + 1) % 10 == 0:
            print(f"  処理済み: {i + 1}/{len(traj[::frame_interval])} フレーム")

    print(f"\n✓ 解析完了: {len(results)} フレームを解析しました\n")

    return pd.DataFrame(results), molecules_history


def plot_analysis_results(df: pd.DataFrame, judgment: Dict, output_dir: str, file_prefix: str = ""):
    """
    拡張プロット（分子種の時間変化と過剰加速判定を含む）

    Args:
        df: 解析結果のDataFrame
        judgment: 過剰加速判定結果
        output_dir: 出力ディレクトリ
        file_prefix: ファイル名プレフィックス（入力ファイル名由来）
    """
    print("=== 拡張グラフ作成中 ===\n")

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.35)

    # 1. HF生成
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['time_ps'], df['n_HF'], 'o-', color='#E63946', linewidth=2, markersize=4)
    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('Number of HF')
    ax1.set_title('HF Generation', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. AlF₃生成
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['time_ps'], df['n_AlF'], 'o-', color='#457B9D', linewidth=2, markersize=4)
    ax2.set_xlabel('Time (ps)')
    ax2.set_ylabel('Number of Al-F bonds')
    ax2.set_title('AlF₃ Formation', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. F貫通
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(df['time_ps'], df['F_penetration_A'], 'o-', color='#06A77D', linewidth=2, markersize=4)
    ax3.set_xlabel('Time (ps)')
    ax3.set_ylabel('Penetration Depth (Å)')
    ax3.set_title('F Penetration', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4-6. 分子種（CF₄, C₂F₄, ベンゼン）
    if 'mol_CF4' in df.columns:
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(df['time_ps'], df['mol_CF4'], 'o-', color='#F4A261', linewidth=2, markersize=4)
        ax4.set_xlabel('Time (ps)')
        ax4.set_ylabel('Number of CF₄')
        title_color = 'red' if df['mol_CF4'].max() > 5 else 'black'
        ax4.set_title('CF₄ Formation (⚠️高温副反応)', fontweight='bold', color=title_color)
        ax4.grid(True, alpha=0.3)

    if 'mol_C2F4' in df.columns:
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(df['time_ps'], df['mol_C2F4'], 'o-', color='#2A9D8F', linewidth=2, markersize=4)
        ax5.set_xlabel('Time (ps)')
        ax5.set_ylabel('Number of C₂F₄')
        ax5.set_title('C₂F₄ Formation', fontweight='bold')
        ax5.grid(True, alpha=0.3)

    if 'mol_benzene_ring' in df.columns:
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(df['time_ps'], df['mol_benzene_ring'], 'o-', color='#E76F51', linewidth=2, markersize=4)
        ax6.set_xlabel('Time (ps)')
        ax6.set_ylabel('Number of Benzene Rings')
        title_color = 'red' if df['mol_benzene_ring'].max() > 0 else 'black'
        ax6.set_title('Aromatic Ring (❌高温特有)', fontweight='bold', color=title_color)
        ax6.grid(True, alpha=0.3)

    # 7. CO/CO2
    if 'mol_CO' in df.columns and 'mol_CO2' in df.columns:
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(df['time_ps'], df['mol_CO'], 'o-', color='#8B4513', linewidth=2, markersize=4, label='CO')
        ax7.plot(df['time_ps'], df['mol_CO2'], 's-', color='#A0522D', linewidth=2, markersize=4, label='CO₂')
        ax7.set_xlabel('Time (ps)')
        ax7.set_ylabel('Number')
        ax7.set_title('CO/CO₂ Formation', fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

    # 8. H2/H2O
    if 'mol_H2' in df.columns and 'mol_H2O' in df.columns:
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(df['time_ps'], df['mol_H2'], 'o-', color='#4682B4', linewidth=2, markersize=4, label='H₂')
        ax8.plot(df['time_ps'], df['mol_H2O'], 's-', color='#5F9EA0', linewidth=2, markersize=4, label='H₂O')
        ax8.set_xlabel('Time (ps)')
        ax8.set_ylabel('Number')
        ax8.set_title('H₂/H₂O Formation', fontweight='bold')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

    # 9. 炭素クラスター
    if 'mol_large_C_cluster' in df.columns:
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.plot(df['time_ps'], df['mol_large_C_cluster'], 'o-', color='#8B4513', linewidth=2, markersize=4)
        ax9.set_xlabel('Time (ps)')
        ax9.set_ylabel('Number of Large C Clusters')
        ax9.set_title('Carbon Clustering (炭化)', fontweight='bold')
        ax9.grid(True, alpha=0.3)

    # 10-11. AlF₃ vs Al還元
    if 'mol_AlF3' in df.columns and 'mol_Al_metal_reduced' in df.columns:
        ax10 = fig.add_subplot(gs[3, 0])
        ax10.plot(df['time_ps'], df['mol_AlF3'], 'o-', color='#264653', linewidth=2, markersize=4, label='AlF₃')
        ax10_twin = ax10.twinx()
        ax10_twin.plot(df['time_ps'], df['mol_Al_metal_reduced'], 's-', color='#E63946', linewidth=2, markersize=4, label='還元Al')
        ax10.set_xlabel('Time (ps)')
        ax10.set_ylabel('AlF₃', color='#264653')
        ax10_twin.set_ylabel('Reduced Al', color='#E63946')
        ax10.set_title('AlF₃ vs Al Reduction', fontweight='bold')
        ax10.grid(True, alpha=0.3)

    # 12. 温度推移
    if 'temperature_K' in df.columns and df['temperature_K'].max() > 0:
        ax11 = fig.add_subplot(gs[3, 1])
        ax11.plot(df['time_ps'], df['temperature_K'], '-', color='#264653', linewidth=1.5)
        ax11.set_xlabel('Time (ps)')
        ax11.set_ylabel('Temperature (K)')
        ax11.set_title('Temperature Evolution', fontweight='bold')
        ax11.grid(True, alpha=0.3)

    # 13. 判定結果サマリー
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.axis('off')

    severity = judgment['severity']
    color_map = {'OK': 'green', 'WARNING': 'orange', 'CRITICAL': 'red'}

    summary_text = f"【過剰加速判定】\n\n"
    summary_text += f"総合判定: {severity}\n\n"

    if judgment['warnings']:
        summary_text += "警告:\n"
        for warning in judgment['warnings'][:5]:  # 最大5つ
            # 絵文字を削除してテキストのみ表示
            clean_warning = warning.replace('⚠️', '').replace('❌', '').replace('✓', '').strip()
            summary_text += f"• {clean_warning}\n"
    else:
        summary_text += "顕著な異常なし\n"

    summary_text += f"\n時間換算:\n"
    summary_text += f"{judgment['time_scale']['T_sim']}K/{judgment['time_scale']['t_sim_ps']:.0f}ps\n"
    summary_text += f"= {judgment['time_scale']['T_exp']}K/{judgment['time_scale']['t_equivalent_hours']:.1f}h\n"

    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor=color_map[severity], alpha=0.3))

    plt.suptitle('拡張解析結果（過剰加速判定含む）', fontsize=16, fontweight='bold', y=0.995)

    plot_filename = f"{file_prefix}_enhanced_analysis.png" if file_prefix else "enhanced_analysis.png"
    output_path = Path(output_dir) / plot_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 拡張グラフを保存: {output_path}\n")


def generate_judgment_report(judgment: Dict, output_dir: str, file_prefix: str = ""):
    """
    過剰加速判定レポートをテキストファイルで出力

    Args:
        judgment: 判定結果の辞書
        output_dir: 出力ディレクトリ
        file_prefix: ファイル名プレフィックス
    """
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("          過剰加速判定レポート")
    report_lines.append("=" * 70)
    report_lines.append("")

    # 総合判定
    severity_symbols = {'OK': '✅', 'WARNING': '⚠️', 'CRITICAL': '❌'}
    report_lines.append(f"【総合判定】 {severity_symbols[judgment['severity']]} {judgment['severity']}")
    report_lines.append("")

    # 警告リスト
    if judgment['warnings']:
        report_lines.append("【検出された問題】")
        for i, warning in enumerate(judgment['warnings'], 1):
            report_lines.append(f"{i}. {warning}")
        report_lines.append("")
    else:
        report_lines.append("【検出された問題】")
        report_lines.append("なし（顕著な異常は検出されませんでした）")
        report_lines.append("")

    # 時間スケール換算
    ts = judgment['time_scale']
    report_lines.append("【時間スケール換算】")
    report_lines.append(f"シミュレーション温度: {ts['T_sim']} K")
    report_lines.append(f"実験相当温度: {ts['T_exp']} K")
    report_lines.append(f"シミュレーション時間: {ts['t_sim_ps']:.1f} ps")
    report_lines.append(f"実験相当時間: {ts['t_equivalent_hours']:.2f} 時間")
    report_lines.append(f"加速率: {ts['acceleration_ratio']:.2e}")
    report_lines.append(f"仮定活性化エネルギー: {ts['Ea_assumed_eV']:.2f} eV")
    report_lines.append("")

    # 最終生成物
    report_lines.append("【最終生成物】")
    for mol, count in judgment['molecules_final'].items():
        symbol = '✅' if count == 0 or mol in ['HF', 'AlF3'] else '⚠️'
        report_lines.append(f"{symbol} {mol}: {count:.1f}")
    report_lines.append("")

    # 反応速度
    report_lines.append("【反応速度定数】")
    for reaction, rate in judgment['rates'].items():
        report_lines.append(f"  {reaction}: {rate:.3f} molecules/ps")
    report_lines.append("")

    # 推奨事項
    report_lines.append("【推奨事項】")
    if judgment['severity'] == 'CRITICAL':
        report_lines.append("❌ この計算結果は定量的評価に使用できません")
        report_lines.append("   → 1000K以下での再計算を強く推奨")
        report_lines.append("   → 複数温度でArrhenius解析が必須")
    elif judgment['severity'] == 'WARNING':
        report_lines.append("⚠️ 注意が必要です")
        report_lines.append("   → 定性的な傾向の把握にとどめる")
        report_lines.append("   → より低温での計算で検証を推奨")
    else:
        report_lines.append("✅ 概ね妥当な範囲と判定されました")
        report_lines.append("   → ただし、より低温での検証は推奨")

    report_lines.append("")
    report_lines.append("=" * 70)

    # ファイル保存
    report_filename = f"{file_prefix}_judgment_report.txt" if file_prefix else "judgment_report.txt"
    report_path = Path(output_dir) / report_filename

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"✓ 判定レポートを保存: {report_path}\n")

    # コンソール出力
    print('\n'.join(report_lines))


def print_summary(df: pd.DataFrame, molecules_final: Dict = None):
    """
    解析結果のサマリーを出力する

    Args:
        df: 解析結果のDataFrame
        molecules_final: 最終フレームの分子種（オプショナル）
    """
    print("=" * 70)
    print("              解析結果サマリー")
    print("=" * 70)

    # HF生成開始時刻
    hf_generation_frames = df[df['n_HF'] > 0]
    if len(hf_generation_frames) > 0:
        first_hf_time = hf_generation_frames.iloc[0]['time_ps']
        print(f"\n✓ HF生成開始時刻: {first_hf_time:.3f} ps (フレーム {hf_generation_frames.iloc[0]['frame']})")
    else:
        print("\n✗ HF生成は観測されませんでした")

    # AlF₃生成開始時刻
    alf_generation_frames = df[df['n_AlF'] > 0]
    if len(alf_generation_frames) > 0:
        first_alf_time = alf_generation_frames.iloc[0]['time_ps']
        print(f"✓ AlF₃生成開始時刻: {first_alf_time:.3f} ps (フレーム {alf_generation_frames.iloc[0]['frame']})")
    else:
        print("✗ AlF₃生成は観測されませんでした")

    # F貫通の最大深度
    max_penetration = df['F_penetration_A'].max()
    if max_penetration > 0:
        print(f"✓ F貫通の最大深度: {max_penetration:.2f} Å")
    else:
        print("✗ F原子のAl層貫通は観測されませんでした")

    # 最終状態
    final_row = df.iloc[-1]
    print(f"\n【最終状態 (t = {final_row['time_ps']:.3f} ps)】")
    print(f"  HF結合数: {final_row['n_HF']:.0f}")
    print(f"  Al-F結合数: {final_row['n_AlF']:.0f}")
    print(f"  O-H結合数: {final_row['n_OH']:.0f}")
    print(f"  F貫通深度: {final_row['F_penetration_A']:.2f} Å")

    # 分子種情報（もし利用可能なら）
    if molecules_final:
        print(f"\n【検出された分子種】")
        for mol, count in molecules_final.items():
            if count > 0:
                print(f"  {mol}: {count:.1f}")

    print("\n" + "=" * 70 + "\n")


# ========================================================================
# メイン実行部
# ========================================================================

def main():
    """メイン実行関数（複数ファイル対応）"""

    print("\n" + "=" * 70)
    print("  Phase 1-A 改良版: 過剰加速判定機能付き解析（複数ファイル対応）")
    print("=" * 70 + "\n")

    # 出力ディレクトリの作成（共通）
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    print(f"出力ディレクトリ: {output_dir}\n")

    # 複数ファイルをループ処理
    for traj_idx, traj_path in enumerate(INPUT_TRAJ_PATHS, 1):
        print("\n" + "=" * 70)
        print(f"  [{traj_idx}/{len(INPUT_TRAJ_PATHS)}] 処理中: {traj_path}")
        print("=" * 70 + "\n")

        # 入力ファイル名に基づいて出力ファイル名のプレフィックスを生成
        file_prefix = generate_output_filename_prefix(traj_path)

        print(f"入力ファイル: {traj_path}")
        print(f"ファイル名プレフィックス: {file_prefix}\n")

        # 拡張Trajectory解析
        df, molecules_history = analyze_trajectory(traj_path, ANALYSIS_PARAMS)

        if df.empty:
            print(f"\n✗ 解析に失敗しました: {traj_path}")
            print("  次のファイルに進みます...\n")
            continue

        # 結果をCSVに保存
        csv_filename = f"{file_prefix}_enhanced_results.csv" if file_prefix else "enhanced_results.csv"
        csv_path = output_dir / csv_filename
        df.to_csv(csv_path, index=False)
        print(f"✓ 解析結果をCSVに保存しました: {csv_path}\n")

        # 過剰加速判定
        molecules_final = molecules_history[-1] if molecules_history else {}

        # ファイル名から温度を推定（1600K以外の場合も対応）
        T_sim = 1600  # デフォルト
        import re
        temp_match = re.search(r'(\d{3,4})K', traj_path)
        if temp_match:
            T_sim = int(temp_match.group(1))

        judgment = judge_excessive_acceleration(molecules_final, df, T_sim=T_sim)

        # グラフ作成
        plot_analysis_results(df, judgment, output_dir, file_prefix)

        # 判定レポート生成
        generate_judgment_report(judgment, output_dir, file_prefix)

        # サマリー出力
        print_summary(df, molecules_final)

        print(f"\n✅ {Path(traj_path).name} の解析完了\n")

    print("\n" + "=" * 70)
    print("  全ファイルの解析完了")
    print("=" * 70 + "\n")

    print("【次のステップ】")
    print("1. 各ファイルの拡張解析グラフ(*_enhanced_analysis.png)を確認してください")
    print("2. 過剰加速判定レポート(*_judgment_report.txt)を確認してください")
    print("3. HF生成とAlF₃生成のタイミングを比較してください")
    print("4. 高温特有の生成物（CF₄、ベンゼン環など）の有無を確認してください")
    print("5. 時間スケール換算の妥当性を評価してください")
    print("\n✓ Phase 1-A 改良版完了\n")


if __name__ == "__main__":
    main()
