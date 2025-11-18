"""
Phase 1-A: 既存データの徹底解析プログラム

目的:
  - Al2O3-PVDF_md_1600K_fix_al.traj を解析
  - HF生成、Al-F結合(AlF₃)生成、F貫通深度を追跡
  - 結果をCSVファイルとグラフで出力

確認事項:
  1. HF生成開始時刻
  2. AlF₃生成位置(表面 vs 界面)
  3. F原子のAl₂O₃層貫通可否
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from ase.io import read
from ase.geometry import get_distances
import warnings
warnings.filterwarnings('ignore')

# ========================================================================
# 設定パラメータ
# ========================================================================

# 入力ファイルのパス (環境に合わせて変更してください)
INPUT_TRAJ_PATH = "Al2O3-PVDF_md_1600K_fix_al.traj"

# 出力ディレクトリ
OUTPUT_DIR = "phase1a_analysis_results"

# 解析パラメータ
ANALYSIS_PARAMS = {
    'HF_cutoff': 1.0,        # HF結合の判定距離 (Å)
    'AlF_cutoff': 2.0,       # Al-F結合の判定距離 (Å)
    'OH_cutoff': 1.1,        # O-H結合の判定距離 (Å)
    'frame_interval': 10,    # 解析するフレーム間隔(全フレーム解析は重いため)
}

# ========================================================================
# 解析関数
# ========================================================================

def analyze_trajectory(traj_path: str, params: Dict) -> pd.DataFrame:
    """
    trajectoryファイルを解析してHF生成、AlF₃生成、F貫通深度を追跡する

    Args:
        traj_path: trajectoryファイルのパス
        params: 解析パラメータの辞書

    Returns:
        解析結果のDataFrame
    """
    print(f"=== Trajectory解析開始: {traj_path} ===\n")

    # trajectoryの読み込み
    try:
        traj = read(traj_path, ":")
        print(f"✓ Trajectoryファイル読み込み完了: {len(traj)} フレーム\n")
    except FileNotFoundError:
        print(f"✗ エラー: ファイルが見つかりません: {traj_path}")
        print("  注: このスクリプトを実際に実行する際は、trajectoryファイルのパスを正しく設定してください。")
        return pd.DataFrame()
    except Exception as e:
        print(f"✗ エラー: ファイルの読み込みに失敗しました: {e}")
        return pd.DataFrame()

    results = []
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

        # Al₂O₃層(tag==2)のAlのみを抽出
        al_oxide_indices = [a.index for a in atoms if a.symbol == 'Al' and hasattr(a, 'tag') and a.tag == 2]
        if not al_oxide_indices:
            # tagが設定されていない場合は、Z座標で判定
            al_positions = atoms.positions[al_indices]
            al_z = al_positions[:, 2]
            z_threshold = np.percentile(al_z, 75)  # 上位25%をAl₂O₃層と仮定
            al_oxide_indices = [al_indices[j] for j, z in enumerate(al_z) if z > z_threshold]

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
        if al_oxide_indices and f_indices:
            al_pos = all_positions[al_oxide_indices]
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
        if f_indices and al_oxide_indices:
            f_z = all_positions[f_indices][:, 2]
            al2o3_surface_z = all_positions[al_oxide_indices][:, 2].max()

            # Al₂O₃表面より下にあるF原子の深度を計算
            penetrating_f = f_z[f_z < al2o3_surface_z]
            if len(penetrating_f) > 0:
                f_penetration = al2o3_surface_z - penetrating_f.min()

        # 5. 温度(もし利用可能なら)
        try:
            temperature = atoms.get_temperature()
        except:
            temperature = 0.0

        # 結果を記録
        results.append({
            'frame': frame_num,
            'time_ps': frame_num * 0.0001,  # 0.1fs/stepと仮定、適宜調整
            'temperature_K': temperature,
            'n_HF': n_hf,
            'n_AlF': n_alf,
            'n_OH': n_oh,
            'F_penetration_A': f_penetration,
        })

        # 進捗表示
        if (i + 1) % 10 == 0:
            print(f"  処理済み: {i + 1}/{len(traj[::frame_interval])} フレーム")

    print(f"\n✓ 解析完了: {len(results)} フレームを解析しました\n")

    return pd.DataFrame(results)


def plot_analysis_results(df: pd.DataFrame, output_dir: str):
    """
    解析結果をプロットして保存する

    Args:
        df: 解析結果のDataFrame
        output_dir: 出力ディレクトリ
    """
    print("=== グラフ作成中 ===\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. HF生成数の時間変化
    ax1 = axes[0, 0]
    ax1.plot(df['time_ps'], df['n_HF'], 'o-', color='#E63946', linewidth=2, markersize=4)
    ax1.set_xlabel('Time (ps)', fontsize=12)
    ax1.set_ylabel('Number of HF bonds', fontsize=12)
    ax1.set_title('HF Generation', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. Al-F結合数の時間変化
    ax2 = axes[0, 1]
    ax2.plot(df['time_ps'], df['n_AlF'], 'o-', color='#457B9D', linewidth=2, markersize=4)
    ax2.set_xlabel('Time (ps)', fontsize=12)
    ax2.set_ylabel('Number of Al-F bonds', fontsize=12)
    ax2.set_title('AlF₃ Formation (Etching)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. F貫通深度の時間変化
    ax3 = axes[1, 0]
    ax3.plot(df['time_ps'], df['F_penetration_A'], 'o-', color='#06A77D', linewidth=2, markersize=4)
    ax3.set_xlabel('Time (ps)', fontsize=12)
    ax3.set_ylabel('F Penetration Depth (Å)', fontsize=12)
    ax3.set_title('F Atom Penetration into Al₂O₃', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. O-H結合数の時間変化
    ax4 = axes[1, 1]
    ax4.plot(df['time_ps'], df['n_OH'], 'o-', color='#F4A261', linewidth=2, markersize=4)
    ax4.set_xlabel('Time (ps)', fontsize=12)
    ax4.set_ylabel('Number of O-H bonds', fontsize=12)
    ax4.set_title('Surface Hydroxylation', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_dir) / "analysis_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ グラフを保存しました: {output_path}\n")

    # 温度プロット(別途)
    if df['temperature_K'].max() > 0:
        fig2, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['time_ps'], df['temperature_K'], '-', color='#264653', linewidth=1.5)
        ax.set_xlabel('Time (ps)', fontsize=12)
        ax.set_ylabel('Temperature (K)', fontsize=12)
        ax.set_title('Temperature Evolution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        temp_path = Path(output_dir) / "temperature_evolution.png"
        plt.savefig(temp_path, dpi=300, bbox_inches='tight')
        print(f"✓ 温度プロットを保存しました: {temp_path}\n")


def print_summary(df: pd.DataFrame):
    """
    解析結果のサマリーを出力する
    """
    print("=" * 60)
    print("              解析結果サマリー")
    print("=" * 60)

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
        print("✗ F原子のAl₂O₃層貫通は観測されませんでした")

    # 最終状態
    final_row = df.iloc[-1]
    print(f"\n【最終状態 (t = {final_row['time_ps']:.3f} ps)】")
    print(f"  HF結合数: {final_row['n_HF']:.0f}")
    print(f"  Al-F結合数: {final_row['n_AlF']:.0f}")
    print(f"  O-H結合数: {final_row['n_OH']:.0f}")
    print(f"  F貫通深度: {final_row['F_penetration_A']:.2f} Å")

    print("\n" + "=" * 60 + "\n")


# ========================================================================
# メイン実行部
# ========================================================================

def main():
    """メイン実行関数"""

    print("\n" + "=" * 60)
    print("  Phase 1-A: 既存データの徹底解析")
    print("=" * 60 + "\n")

    # 出力ディレクトリの作成
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    print(f"出力ディレクトリ: {output_dir}\n")

    # Trajectory解析
    df = analyze_trajectory(INPUT_TRAJ_PATH, ANALYSIS_PARAMS)

    if df.empty:
        print("\n✗ 解析に失敗しました。プログラムを終了します。")
        print("\n【次のステップ】")
        print("1. INPUT_TRAJ_PATH を正しいtrajectoryファイルのパスに設定してください")
        print("2. trajectoryファイルが存在することを確認してください")
        return

    # 結果をCSVに保存
    csv_path = output_dir / "analysis_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ 解析結果をCSVに保存しました: {csv_path}\n")

    # グラフ作成
    plot_analysis_results(df, output_dir)

    # サマリー出力
    print_summary(df)

    print("【次のステップ】")
    print("1. 解析結果グラフ(analysis_results.png)を確認してください")
    print("2. HF生成とAlF₃生成のタイミングを比較してください")
    print("3. F貫通深度が有意かどうかを評価してください")
    print("\n✓ Phase 1-A 完了\n")


if __name__ == "__main__":
    main()
