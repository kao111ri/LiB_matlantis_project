#!/usr/bin/env python3
"""
Al-NMC界面解析スクリプト（混入密度ベース）

界面の断面積で正規化した「界面混入密度」を使用してスコア計算・可視化を行う。

使用方法:
    python al_nmc_density_analysis.py [入力CSVパス]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from pathlib import Path

# 日本語フォント対応（利用可能な場合）
try:
    import japanize_matplotlib
except ImportError:
    print("警告: japanize_matplotlibがインストールされていません。日本語が表示できない場合があります。")


# ============================================================================
# 設定・定数
# ============================================================================

# 基板の断面積 (Å²)
AREA_MAP = {
    'Al2O3': 13.94673902 * 13.08364686,   # ~182.5 Å²
    'AlF3':  11.76094577 * 12.80689447,   # ~150.6 Å²
    'Al':    16.952258347397713 * 16.952258350558857,  # ~287.4 Å²
}

# 基板を構成する元素（上層への混入をカウント）
SUBSTRATE_ELEMENTS = ['Al']

# カソードを構成する元素（下層への混入をカウント）
CATHODE_ELEMENTS = ['Li', 'Mn', 'Co', 'Ni']

# スコア計算の重み設定
METRICS_CONFIG = {
    'tensile_strength_GPa':              {'weight': 0.2, 'high_is_good': False},
    'fracture_step':                     {'weight': 0.0, 'high_is_good': False},
    'interfacial_contamination_density': {'weight': 0.8, 'high_is_good': False},
}

# 可視化の色設定
COLOR_MAP = {
    'Co': '#1f77b4',
    'Mn': '#ff7f0e',
    'Co3O4': '#2ca02c',
    'NiO': '#d62728',
    'Li2O': '#9467bd',
    'Carbon': '#8c564b',
    'graphite': '#e377c2',
    'Li3MnCoNiO6': '#bcbd22',
    'Mn3O4': '#17becf',
}

# フォントサイズ設定
TITLE_FS = 20
LABEL_FS = 18
TICK_FS = 16
LEGEND_FS = 16
LEGEND_TITLE_FS = 18


# ============================================================================
# ユーティリティ関数
# ============================================================================

def format_chemical_formula(text):
    """
    文字列内の数値をLaTeXの下付き文字形式に変換する。
    例: 'Al2O3' -> 'Al$_{2}$O$_{3}$'
    """
    if not isinstance(text, str) or '$' in text:
        return text
    return re.sub(r'(\d+)', r'$_{\1}$', text)


def normalize_metric(series, high_is_good=True):
    """
    PandasのSeriesをミンマックス正規化する。

    Args:
        series: 正規化するPandas Series
        high_is_good: Trueなら高い値が良い、Falseなら低い値が良い

    Returns:
        正規化されたSeries (0-1の範囲)
    """
    series_finite = series.replace([np.inf, -np.inf], np.nan).fillna(0)
    min_val = series_finite.min()
    max_val = series_finite.max()

    if max_val - min_val == 0:
        return pd.Series(0.5, index=series.index)

    if high_is_good:
        return (series_finite - min_val) / (max_val - min_val)
    else:
        return (max_val - series_finite) / (max_val - min_val)


# ============================================================================
# スコア計算モジュール
# ============================================================================

def calculate_interfacial_density(df):
    """
    界面の断面積で規格化された「界面混入密度」を計算する。

    計算される列:
        - cross_sectional_area_A2: 断面積 (Å²)
        - substrate_mixed_atoms: 基板から上層への混入原子数
        - cathode_mixed_atoms: カソードから下層への混入原子数
        - total_mixed_atoms: 合計混入原子数
        - substrate_contamination_density: 基板混入密度 (atoms/Å²)
        - cathode_contamination_density: カソード混入密度 (atoms/Å²)
        - interfacial_contamination_density: 全体混入密度 (atoms/Å²)
    """
    # ステップ1: 断面積を割り当て
    df['cross_sectional_area_A2'] = df['substrate'].map(AREA_MAP)

    if df['cross_sectional_area_A2'].isnull().any():
        unknown = df[df['cross_sectional_area_A2'].isnull()]['substrate'].unique()
        print(f"警告: 未知の基板が見つかりました: {unknown}")
        print("AREA_MAPに定義を追加してください。")
        df['cross_sectional_area_A2'].fillna(0, inplace=True)

    # ステップ2: 混入原子数を計算
    # 基板からの混入（上層に移動した基板原子）
    df['substrate_mixed_atoms'] = 0
    for elem in SUBSTRATE_ELEMENTS:
        col = f'{elem}_upper'
        if col in df.columns:
            df['substrate_mixed_atoms'] += df[col].fillna(0)

    # カソードからの混入（下層に移動したカソード原子）
    df['cathode_mixed_atoms'] = 0
    for elem in CATHODE_ELEMENTS:
        col = f'{elem}_lower'
        if col in df.columns:
            df['cathode_mixed_atoms'] += df[col].fillna(0)

    df['total_mixed_atoms'] = df['substrate_mixed_atoms'] + df['cathode_mixed_atoms']

    # ステップ3: 混入密度を計算
    valid_mask = df['cross_sectional_area_A2'] > 0

    df['substrate_contamination_density'] = 0.0
    df['cathode_contamination_density'] = 0.0
    df['interfacial_contamination_density'] = 0.0

    df.loc[valid_mask, 'substrate_contamination_density'] = (
        df.loc[valid_mask, 'substrate_mixed_atoms'] / df.loc[valid_mask, 'cross_sectional_area_A2']
    )
    df.loc[valid_mask, 'cathode_contamination_density'] = (
        df.loc[valid_mask, 'cathode_mixed_atoms'] / df.loc[valid_mask, 'cross_sectional_area_A2']
    )
    df.loc[valid_mask, 'interfacial_contamination_density'] = (
        df.loc[valid_mask, 'total_mixed_atoms'] / df.loc[valid_mask, 'cross_sectional_area_A2']
    )

    return df


def calculate_scores(df, metrics_config=None):
    """
    総合スコアを計算する。

    Args:
        df: 入力DataFrame
        metrics_config: 評価指標の設定辞書（Noneの場合はデフォルト使用）

    Returns:
        スコア計算済みのDataFrame
    """
    if metrics_config is None:
        metrics_config = METRICS_CONFIG

    # 混入密度を計算
    df = calculate_interfacial_density(df)

    # スコア計算
    df['総合スコア'] = 0.0

    print("\n評価指標と方針:")
    for metric, config in metrics_config.items():
        if config['weight'] == 0:
            continue

        direction = "低いほど高評価" if not config['high_is_good'] else "高いほど高評価"
        print(f"  - {metric}: 重み {config['weight']*100:.0f}%, ({direction})")

        if metric not in df.columns:
            print(f"    警告: '{metric}' がデータに存在しないためスキップ")
            continue

        df[metric] = df[metric].fillna(0)

        norm_col = f'norm_{metric}'
        df[norm_col] = normalize_metric(df[metric], high_is_good=config['high_is_good'])

        df['総合スコア'] += df[norm_col] * config['weight']

    df['総合スコア'] *= 100

    return df


# ============================================================================
# 可視化モジュール
# ============================================================================

def plot_scoring_results(df, output_dir):
    """
    スコアリング結果を可視化する。

    生成されるグラフ:
        1. 混入密度散布図
        2. 基板別スコア箱ひげ図
        3. スコア分布（swarm plot）
        4. カソード混入密度分布
        5. 基板混入密度分布
        6. グループ別平均スコア棒グラフ
        7. カソード材料別スコア分布
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nグラフ作成を開始...")

    # 化学式フォーマット
    df['substrate_formatted'] = df['substrate'].apply(format_chemical_formula)
    df['material_formatted'] = df['material'].apply(format_chemical_formula)

    sns.set_theme(style="whitegrid")

    # --- 1. 混入密度散布図 ---
    plt.figure(figsize=(8, 6.5))
    sns.scatterplot(
        data=df,
        x='substrate_contamination_density',
        y='cathode_contamination_density',
        style='substrate_formatted',
        hue='material',
        s=200,
        alpha=0.8,
        palette=COLOR_MAP
    )
    plt.title('Substrate vs. Cathode Contamination Density', fontsize=TITLE_FS, weight='bold')
    plt.xlabel('Substrate Contamination Density (atoms/Å²)', fontsize=LABEL_FS)
    plt.ylabel('Cathode Contamination Density (atoms/Å²)', fontsize=LABEL_FS)
    plt.xticks(fontsize=TICK_FS)
    plt.yticks(fontsize=TICK_FS)

    handles, labels = plt.gca().get_legend_handles_labels()
    formatted_labels = [format_chemical_formula(label) for label in labels]
    plt.legend(handles, formatted_labels, bbox_to_anchor=(1.02, 1), loc='upper left',
               fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS, title='Legend')

    plt.tight_layout(rect=[0, 0, 0.80, 1])
    path1 = output_dir / "density_scatter_plot.png"
    plt.savefig(path1, dpi=300)
    print(f"  保存: {path1}")
    plt.close()

    # --- 2. 基板別スコア箱ひげ図 ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='substrate_formatted', y='総合スコア', data=df)
    plt.title('Total Score by Al Surface', fontsize=TITLE_FS, weight='bold')
    plt.xlabel('Al Surface Condition', fontsize=LABEL_FS)
    plt.ylabel('Total Score', fontsize=LABEL_FS)
    plt.xticks(fontsize=TICK_FS)
    plt.yticks(fontsize=TICK_FS)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    path2 = output_dir / "total_score_by_surface_boxplot.png"
    plt.savefig(path2, dpi=300)
    print(f"  保存: {path2}")
    plt.close()

    # --- 3. スコア分布 swarm plot ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='substrate_formatted', y='総合スコア', data=df, color='lightgray', showfliers=False)
    sns.swarmplot(x='substrate_formatted', y='総合スコア', hue='material', data=df,
                  size=8, palette=COLOR_MAP)
    plt.title('Score Distribution', fontsize=TITLE_FS, weight='bold')
    plt.xlabel('Al Surface Condition', fontsize=LABEL_FS)
    plt.ylabel('Total Score', fontsize=LABEL_FS)
    plt.xticks(fontsize=TICK_FS)
    plt.yticks(fontsize=TICK_FS)

    handles, labels = plt.gca().get_legend_handles_labels()
    formatted_labels = [format_chemical_formula(label) for label in labels]
    plt.legend(handles, formatted_labels, title='Cathode Material',
               fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS)

    plt.tight_layout()
    path3 = output_dir / "score_distribution_swarm_plot.png"
    plt.savefig(path3, dpi=300)
    print(f"  保存: {path3}")
    plt.close()

    # --- 4. カソード混入密度分布 ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='substrate_formatted', y='cathode_contamination_density', data=df,
                color='lightgray', showfliers=False)
    sns.swarmplot(x='substrate_formatted', y='cathode_contamination_density', hue='material', data=df,
                  size=8, palette=COLOR_MAP)
    plt.title('Cathode Contamination Density Distribution', fontsize=TITLE_FS, weight='bold')
    plt.xlabel('Al Surface Condition', fontsize=LABEL_FS)
    plt.ylabel('Cathode Contamination Density (atoms/Å²)', fontsize=LABEL_FS)
    plt.xticks(fontsize=TICK_FS)
    plt.yticks(fontsize=TICK_FS)

    handles, labels = plt.gca().get_legend_handles_labels()
    formatted_labels = [format_chemical_formula(label) for label in labels]
    plt.legend(handles, formatted_labels, title='Cathode Material',
               fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS)

    plt.tight_layout()
    path4 = output_dir / "cathode_density_swarm_plot.png"
    plt.savefig(path4, dpi=300)
    print(f"  保存: {path4}")
    plt.close()

    # --- 5. 基板混入密度分布 ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='substrate_formatted', y='substrate_contamination_density', data=df,
                color='lightgray', showfliers=False)
    sns.swarmplot(x='substrate_formatted', y='substrate_contamination_density', hue='material', data=df,
                  size=8, palette=COLOR_MAP)
    plt.title('Substrate Contamination Density Distribution', fontsize=TITLE_FS, weight='bold')
    plt.xlabel('Al Surface Condition', fontsize=LABEL_FS)
    plt.ylabel('Substrate Contamination Density (atoms/Å²)', fontsize=LABEL_FS)
    plt.xticks(fontsize=TICK_FS)
    plt.yticks(fontsize=TICK_FS)

    handles, labels = plt.gca().get_legend_handles_labels()
    formatted_labels = [format_chemical_formula(label) for label in labels]
    plt.legend(handles, formatted_labels, title='Cathode Material',
               fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS)

    plt.tight_layout()
    path5 = output_dir / "substrate_density_swarm_plot.png"
    plt.savefig(path5, dpi=300)
    print(f"  保存: {path5}")
    plt.close()

    # --- 6. グループ別平均スコア棒グラフ ---
    g = sns.catplot(
        data=df, kind="bar", x="substrate_formatted", y="総合スコア",
        hue="material", palette=COLOR_MAP,
        height=6, aspect=1.2,
        legend=False
    )
    g.ax.set_title('Avg. Score by Surface & Cathode', fontsize=TITLE_FS, weight='bold')
    g.set_axis_labels('Al Surface Condition', 'Average Total Score')

    for ax in g.axes.flat:
        ax.set_xlabel(ax.get_xlabel(), fontsize=LABEL_FS)
        ax.set_ylabel(ax.get_ylabel(), fontsize=LABEL_FS)
        ax.tick_params(axis='x', labelsize=TICK_FS)
        ax.tick_params(axis='y', labelsize=TICK_FS)

    handles, labels = g.ax.get_legend_handles_labels()
    formatted_labels = [format_chemical_formula(label) for label in labels]
    g.ax.legend(handles, formatted_labels, title='Cathode Material', bbox_to_anchor=(1.02, 1),
                loc='upper left', title_fontsize=LEGEND_TITLE_FS, fontsize=LEGEND_FS)

    g.ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    path6 = output_dir / "total_score_by_group_barplot.png"
    g.savefig(path6, dpi=300)
    print(f"  保存: {path6}")
    plt.close()

    # --- 7. カソード材料別スコア分布 ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='material_formatted', y='総合スコア', palette=COLOR_MAP)
    sns.swarmplot(data=df, x='material_formatted', y='総合スコア', color=".25", alpha=0.7)
    plt.title('Score Distribution by Cathode', fontsize=TITLE_FS, weight='bold')
    plt.xlabel('Cathode Material', fontsize=LABEL_FS)
    plt.ylabel('Total Score', fontsize=LABEL_FS)
    plt.xticks(fontsize=TICK_FS - 2, rotation=45, ha='right')
    plt.yticks(fontsize=TICK_FS)
    plt.tight_layout()
    path7 = output_dir / "total_score_by_material_boxplot.png"
    plt.savefig(path7, dpi=300)
    print(f"  保存: {path7}")
    plt.close()

    print(f"\n全{7}グラフを {output_dir} に保存しました。")


def plot_pressure_vs_score(df, output_dir):
    """
    圧力とスコアの関係をファセット散布図で可視化する。
    """
    output_dir = Path(output_dir)

    g = sns.relplot(
        data=df,
        x='pressure_GPa',
        y='総合スコア',
        col='substrate',
        row='material',
        hue='high_temp_K',
        palette='viridis',
        kind='scatter',
        height=3,
        aspect=1.2
    )

    g.set_axis_labels("Pressure (GPa)", "Total Score")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    plt.suptitle('Pressure vs. Score (by Composition & Temperature)', y=1.03, fontsize=16, weight='bold')
    plt.tight_layout()

    path = output_dir / "pressure_vs_score_by_composition.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  保存: {path}")
    plt.close()


# ============================================================================
# メイン関数
# ============================================================================

def run_analysis(csv_path, output_dir="./output/density_analysis"):
    """
    解析のメイン関数。

    Args:
        csv_path: 入力CSVファイルのパス
        output_dir: 出力ディレクトリ
    """
    print("=" * 60)
    print("Al-NMC界面解析（混入密度ベース）")
    print("=" * 60)

    # データ読み込み
    try:
        df = pd.read_csv(csv_path)
        print(f"\n入力ファイル: {csv_path}")
        print(f"データ数: {len(df)} 行")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません '{csv_path}'")
        return None

    # スコア計算
    print("\n--- スコア計算 ---")
    df = calculate_scores(df)

    # 結果のソート
    df_sorted = df.sort_values(by='総合スコア', ascending=False)

    # 結果表示
    display_cols = ['substrate', 'material', 'pressure_GPa', 'high_temp_K', '総合スコア',
                    'interfacial_contamination_density', 'tensile_strength_GPa']
    display_cols = [c for c in display_cols if c in df_sorted.columns]

    print("\n--- スコアリング結果 トップ10 ---")
    print(df_sorted[display_cols].head(10).round(3).to_string())

    print("\n--- スコアリング結果 下位10 ---")
    print(df_sorted[display_cols].tail(10).round(3).to_string())

    # 結果保存
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "interfacial_density_scored_results.csv"
    df_sorted.to_csv(output_csv, index=False)
    print(f"\n結果を保存: {output_csv}")

    # 可視化
    plot_scoring_results(df_sorted, output_dir)

    # 圧力vs スコア（列が存在する場合）
    if 'pressure_GPa' in df_sorted.columns and 'high_temp_K' in df_sorted.columns:
        plot_pressure_vs_score(df_sorted, output_dir)

    print("\n" + "=" * 60)
    print("解析完了")
    print("=" * 60)

    return df_sorted


if __name__ == "__main__":
    import sys

    # デフォルトの入力パス
    default_csv = "./output/comprehensive_analysis_output/comprehensive_analysis_results.csv"

    csv_path = sys.argv[1] if len(sys.argv) > 1 else default_csv

    run_analysis(csv_path)
