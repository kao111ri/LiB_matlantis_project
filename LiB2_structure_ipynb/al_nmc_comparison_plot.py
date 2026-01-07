#!/usr/bin/env python3
"""
Al-NMC界面解析 - 引張強度と混入密度の比較グラフ

基板材料（Al, Al2O3, AlF3）ごとに引張強度と混入密度を
横並び棒グラフで可視化する。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 日本語フォント対応
try:
    import japanize_matplotlib
except ImportError:
    pass


# ============================================================================
# 設定・定数
# ============================================================================

# 基板の断面積 (Å²)
AREA_MAP = {
    'Al2O3': 13.94673902 * 13.08364686,   # ~182.5 Å²
    'AlF3':  11.76094577 * 12.80689447,   # ~150.6 Å²
    'Al':    16.952258347397713 * 16.952258350558857,  # ~287.4 Å²
}

# 基板を構成する元素
SUBSTRATE_ELEMENTS = ['Al']

# カソードを構成する元素
CATHODE_ELEMENTS = ['Li', 'Mn', 'Co', 'Ni']


# ============================================================================
# データ処理
# ============================================================================

def calculate_contamination_density(df):
    """
    混入密度を計算する。
    """
    df = df.copy()

    # 断面積を割り当て
    df['cross_sectional_area_A2'] = df['substrate'].map(AREA_MAP)
    df['cross_sectional_area_A2'].fillna(0, inplace=True)

    # 基板からの混入原子数
    df['substrate_mixed_atoms'] = 0
    for elem in SUBSTRATE_ELEMENTS:
        col = f'{elem}_upper'
        if col in df.columns:
            df['substrate_mixed_atoms'] += df[col].fillna(0)

    # カソードからの混入原子数
    df['cathode_mixed_atoms'] = 0
    for elem in CATHODE_ELEMENTS:
        col = f'{elem}_lower'
        if col in df.columns:
            df['cathode_mixed_atoms'] += df[col].fillna(0)

    df['total_mixed_atoms'] = df['substrate_mixed_atoms'] + df['cathode_mixed_atoms']

    # 混入密度を計算
    valid_mask = df['cross_sectional_area_A2'] > 0
    df['total_contamination_density'] = 0.0
    df.loc[valid_mask, 'total_contamination_density'] = (
        df.loc[valid_mask, 'total_mixed_atoms'] / df.loc[valid_mask, 'cross_sectional_area_A2']
    )

    return df


def create_summary_data(df, group_by='substrate'):
    """
    グループごとの平均値を計算してサマリーデータを作成する。

    Args:
        df: 入力DataFrame
        group_by: グループ化する列名（'substrate' or 'material'）

    Returns:
        サマリーDataFrame
    """
    # 混入密度を計算
    df = calculate_contamination_density(df)

    # グループごとの平均を計算
    summary = df.groupby(group_by).agg({
        'tensile_strength_GPa': 'mean',
        'total_contamination_density': 'mean',
    }).reset_index()

    return summary


def create_filtered_summary(df, conditions=None):
    """
    条件でフィルタリングしてサマリーを作成する。

    Args:
        df: 入力DataFrame
        conditions: フィルタ条件の辞書
            例: {'material': 'Li3MnCoNiO6', 'high_temp_K': 500}

    Returns:
        サマリーDataFrame
    """
    df_filtered = df.copy()

    # 条件でフィルタリング
    if conditions:
        for col, val in conditions.items():
            if col in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[col] == val]

    return create_summary_data(df_filtered, group_by='substrate')


# ============================================================================
# 可視化
# ============================================================================

def plot_comparison_bar(summary_df, title_suffix="", output_path=None):
    """
    引張強度と混入密度の2軸横並び棒グラフを作成する。

    Args:
        summary_df: サマリーDataFrame（substrate, tensile_strength_GPa, total_contamination_density）
        title_suffix: タイトルに追加するテキスト
        output_path: 保存先パス（Noneの場合は表示のみ）
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    materials = summary_df['substrate'].tolist()
    tensile = summary_df['tensile_strength_GPa'].tolist()
    density = summary_df['total_contamination_density'].tolist()

    x = np.arange(len(materials))
    width = 0.35

    # 左軸: 引張強度（青）
    color1 = '#1f77b4'
    bars1 = ax1.bar(x - width/2, tensile, width, label='Tensile Strength (GPa)', color=color1)
    ax1.set_xlabel('Material Comparison', fontsize=14)
    ax1.set_ylabel('Tensile Strength (GPa)', fontsize=14, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(materials, fontsize=12)

    # 右軸: 混入密度（オレンジ）
    ax2 = ax1.twinx()
    color2 = '#ff7f0e'
    bars2 = ax2.bar(x + width/2, density, width, label='Total Contamination Density', color=color2)
    ax2.set_ylabel('Total Contamination Density', fontsize=14, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # タイトル
    title = "Material Comparison"
    if title_suffix:
        title += f"\n({title_suffix})"
    plt.title(title, fontsize=16, weight='bold')

    # グリッド
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"保存: {output_path}")

    plt.show()
    plt.close()

    return fig


def plot_by_conditions(df, material=None, high_temp_K=None, pressure_GPa=None, output_dir=None):
    """
    指定した条件でフィルタリングしてグラフを作成する。

    Args:
        df: 入力DataFrame
        material: カソード材料でフィルタ
        high_temp_K: 高温条件でフィルタ
        pressure_GPa: 圧力でフィルタ
        output_dir: 出力ディレクトリ
    """
    conditions = {}
    title_parts = []

    if material:
        conditions['material'] = material
        title_parts.append(f"Cell: {material}")
    if high_temp_K:
        conditions['high_temp_K'] = high_temp_K
        title_parts.append(f"Temp: {high_temp_K}K")
    if pressure_GPa:
        conditions['pressure_GPa'] = pressure_GPa
        title_parts.append(f"P: {pressure_GPa}GPa")

    summary = create_filtered_summary(df, conditions)

    if len(summary) == 0:
        print("警告: 指定条件に該当するデータがありません")
        return None

    title_suffix = ", ".join(title_parts) if title_parts else "All Data"

    output_path = None
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"comparison_{'_'.join(str(v) for v in conditions.values()) if conditions else 'all'}.png"
        output_path = output_dir / filename

    return plot_comparison_bar(summary, title_suffix, output_path)


# ============================================================================
# メイン関数
# ============================================================================

def run_analysis(csv_path, output_dir="./output/comparison_plots"):
    """
    解析のメイン関数。

    Args:
        csv_path: 入力CSVファイルのパス
        output_dir: 出力ディレクトリ
    """
    print("=" * 60)
    print("Al-NMC界面解析 - 引張強度と混入密度の比較")
    print("=" * 60)

    # データ読み込み
    try:
        df = pd.read_csv(csv_path)
        print(f"\n入力ファイル: {csv_path}")
        print(f"データ数: {len(df)} 行")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません '{csv_path}'")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 混入密度を計算
    df = calculate_contamination_density(df)

    # サマリーCSVを出力
    summary_all = create_summary_data(df, group_by='substrate')
    print("\n--- 基板別サマリー（全データ平均） ---")
    print(summary_all.round(4).to_string(index=False))

    summary_csv = output_dir / "material_comparison_summary.csv"
    summary_all.to_csv(summary_csv, index=False)
    print(f"\nサマリー保存: {summary_csv}")

    # 全データの比較グラフ
    plot_comparison_bar(
        summary_all,
        title_suffix="All Data Average",
        output_path=output_dir / "comparison_all_average.png"
    )

    # 条件別グラフの例（material と high_temp_K の組み合わせ）
    if 'material' in df.columns and 'high_temp_K' in df.columns:
        materials = df['material'].unique()
        temps = df['high_temp_K'].unique()

        print("\n--- 条件別グラフ作成 ---")
        for mat in materials[:3]:  # 最初の3つの材料のみ
            for temp in temps[:2]:  # 最初の2つの温度のみ
                conditions = {'material': mat, 'high_temp_K': temp}
                summary = create_filtered_summary(df, conditions)
                if len(summary) > 0:
                    title_suffix = f"Cell: {mat}, Temp: {temp}K"
                    filename = f"comparison_{mat}_{temp}K.png"
                    plot_comparison_bar(
                        summary,
                        title_suffix=title_suffix,
                        output_path=output_dir / filename
                    )

    # 詳細データもCSV出力
    detail_csv = output_dir / "contamination_density_detail.csv"
    df.to_csv(detail_csv, index=False)
    print(f"詳細データ保存: {detail_csv}")

    print("\n" + "=" * 60)
    print("解析完了")
    print("=" * 60)

    return df


if __name__ == "__main__":
    import sys

    default_csv = "./output/comprehensive_analysis_output/comprehensive_analysis_results.csv"
    csv_path = sys.argv[1] if len(sys.argv) > 1 else default_csv

    run_analysis(csv_path)
