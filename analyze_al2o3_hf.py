#!/usr/bin/env python3
"""
Al2O3_HF.ipynbで行ったエッチング反応シミュレーションの解析プログラム

解析内容:
- Al-F結合数の時間変化（エッチング進行）
- O-H結合数の時間変化（表面水酸基化）
- H2O分子生成の追跡
- 複数温度条件の比較
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional

# 日本語フォント設定（オプション）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Al2O3EtchingAnalyzer:
    """Al2O3エッチング反応の解析クラス"""

    def __init__(self, data_dir: str = "validation_etching"):
        """
        Parameters
        ----------
        data_dir : str
            解析対象のデータディレクトリ
        """
        self.data_dir = Path(data_dir)
        self.data: Dict[str, pd.DataFrame] = {}
        self.temperatures: List[float] = []

    def load_data(self, pattern: str = "etching_test_*K_etching.log") -> None:
        """
        ログファイルを読み込む

        Parameters
        ----------
        pattern : str
            ファイル名のパターン
        """
        log_files = sorted(self.data_dir.glob(pattern))

        if not log_files:
            print(f"警告: {self.data_dir}/{pattern} に該当するファイルが見つかりません")
            return

        print(f"--- データ読み込み開始 ---")
        for log_file in log_files:
            # ファイル名から温度を抽出 (例: etching_test_350K_etching.log -> 350)
            temp_str = log_file.stem.split('_')[2].replace('K', '')
            temp = float(temp_str)

            # CSVとして読み込み
            df = pd.read_csv(log_file)
            self.data[f"{temp}K"] = df
            self.temperatures.append(temp)

            print(f"  読み込み完了: {log_file.name} ({len(df)} ステップ)")

        self.temperatures = sorted(self.temperatures)
        print(f"--- 読み込み完了: {len(self.data)} ファイル ---\n")

    def plot_time_series(self, save_fig: bool = True, output_dir: str = "analysis_results") -> None:
        """
        時系列プロットを作成

        Parameters
        ----------
        save_fig : bool
            図を保存するかどうか
        output_dir : str
            保存先ディレクトリ
        """
        if not self.data:
            print("エラー: データが読み込まれていません")
            return

        # 出力ディレクトリの作成
        if save_fig:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

        # 3つのサブプロットを作成
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle('Al2O3 Etching Reaction Analysis', fontsize=16, fontweight='bold')

        colors = plt.cm.viridis(np.linspace(0, 1, len(self.data)))

        for idx, (temp_label, df) in enumerate(self.data.items()):
            color = colors[idx]

            # Al-F結合数（エッチング）
            axes[0].plot(df['time_ps'], df['n_AlF_bonds'],
                        label=temp_label, color=color, linewidth=2, marker='o', markersize=4)

            # O-H結合数（水酸基化）
            axes[1].plot(df['time_ps'], df['n_OH_bonds'],
                        label=temp_label, color=color, linewidth=2, marker='s', markersize=4)

            # H2O分子数
            axes[2].plot(df['time_ps'], df['n_H2O_mols'],
                        label=temp_label, color=color, linewidth=2, marker='^', markersize=4)

        # 軸ラベルと凡例の設定
        axes[0].set_ylabel('Al-F Bonds', fontsize=12, fontweight='bold')
        axes[0].set_title('Etching Progress (Al-F Bond Formation)', fontsize=12)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)

        axes[1].set_ylabel('O-H Bonds', fontsize=12, fontweight='bold')
        axes[1].set_title('Surface Hydroxylation (O-H Bond Formation)', fontsize=12)
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)

        axes[2].set_xlabel('Time (ps)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('H2O Molecules', fontsize=12, fontweight='bold')
        axes[2].set_title('Water Formation', fontsize=12)
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_fig:
            fig_path = output_path / "al2o3_etching_time_series.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"図を保存しました: {fig_path}")

        plt.show()

    def calculate_statistics(self) -> pd.DataFrame:
        """
        統計情報を計算

        Returns
        -------
        pd.DataFrame
            統計情報をまとめたデータフレーム
        """
        if not self.data:
            print("エラー: データが読み込まれていません")
            return pd.DataFrame()

        stats_list = []

        for temp_label, df in self.data.items():
            stats = {
                'Temperature': temp_label,
                'AlF_final': df['n_AlF_bonds'].iloc[-1],
                'AlF_max': df['n_AlF_bonds'].max(),
                'AlF_mean': df['n_AlF_bonds'].mean(),
                'OH_final': df['n_OH_bonds'].iloc[-1],
                'OH_max': df['n_OH_bonds'].max(),
                'OH_mean': df['n_OH_bonds'].mean(),
                'H2O_final': df['n_H2O_mols'].iloc[-1],
                'H2O_max': df['n_H2O_mols'].max(),
                'H2O_mean': df['n_H2O_mols'].mean(),
                'Simulation_time_ps': df['time_ps'].iloc[-1]
            }
            stats_list.append(stats)

        stats_df = pd.DataFrame(stats_list)
        return stats_df

    def plot_temperature_dependence(self, save_fig: bool = True, output_dir: str = "analysis_results") -> None:
        """
        温度依存性のプロット

        Parameters
        ----------
        save_fig : bool
            図を保存するかどうか
        output_dir : str
            保存先ディレクトリ
        """
        if not self.data:
            print("エラー: データが読み込まれていません")
            return

        stats_df = self.calculate_statistics()

        # 出力ディレクトリの作成
        if save_fig:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

        # 温度を数値に変換
        temps = [float(t.replace('K', '')) for t in stats_df['Temperature']]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Temperature Dependence of Etching Reactions', fontsize=16, fontweight='bold')

        # Al-F結合の温度依存性
        axes[0].plot(temps, stats_df['AlF_final'], 'o-', markersize=10, linewidth=2, label='Final')
        axes[0].plot(temps, stats_df['AlF_max'], 's--', markersize=8, linewidth=2, label='Max')
        axes[0].set_xlabel('Temperature (K)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Al-F Bonds', fontsize=12, fontweight='bold')
        axes[0].set_title('Etching Progress', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # O-H結合の温度依存性
        axes[1].plot(temps, stats_df['OH_final'], 'o-', markersize=10, linewidth=2, label='Final')
        axes[1].plot(temps, stats_df['OH_max'], 's--', markersize=8, linewidth=2, label='Max')
        axes[1].set_xlabel('Temperature (K)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('O-H Bonds', fontsize=12, fontweight='bold')
        axes[1].set_title('Surface Hydroxylation', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # H2O分子の温度依存性
        axes[2].plot(temps, stats_df['H2O_final'], 'o-', markersize=10, linewidth=2, label='Final')
        axes[2].plot(temps, stats_df['H2O_max'], 's--', markersize=8, linewidth=2, label='Max')
        axes[2].set_xlabel('Temperature (K)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('H2O Molecules', fontsize=12, fontweight='bold')
        axes[2].set_title('Water Formation', fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_fig:
            fig_path = output_path / "al2o3_etching_temperature_dependence.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"図を保存しました: {fig_path}")

        plt.show()

    def generate_report(self, output_dir: str = "analysis_results") -> None:
        """
        解析レポートを生成

        Parameters
        ----------
        output_dir : str
            保存先ディレクトリ
        """
        if not self.data:
            print("エラー: データが読み込まれていません")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        stats_df = self.calculate_statistics()

        # CSVファイルとして保存
        csv_path = output_path / "al2o3_etching_statistics.csv"
        stats_df.to_csv(csv_path, index=False)
        print(f"統計情報を保存しました: {csv_path}")

        # テキストレポートの生成
        report_path = output_path / "al2o3_etching_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("Al2O3 エッチング反応解析レポート\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"解析日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"データディレクトリ: {self.data_dir}\n")
            f.write(f"解析ファイル数: {len(self.data)}\n\n")

            f.write("-" * 70 + "\n")
            f.write("統計情報\n")
            f.write("-" * 70 + "\n\n")
            f.write(stats_df.to_string(index=False))
            f.write("\n\n")

            f.write("-" * 70 + "\n")
            f.write("考察\n")
            f.write("-" * 70 + "\n\n")

            for temp_label, df in self.data.items():
                alf_final = df['n_AlF_bonds'].iloc[-1]
                oh_final = df['n_OH_bonds'].iloc[-1]
                h2o_final = df['n_H2O_mols'].iloc[-1]

                f.write(f"【{temp_label}】\n")
                f.write(f"  - Al-F結合数（最終）: {alf_final}\n")
                f.write(f"  - O-H結合数（最終）: {oh_final}\n")
                f.write(f"  - H2O分子数（最終）: {h2o_final}\n")

                if alf_final > 0:
                    f.write(f"  → エッチング反応が進行しています\n")
                else:
                    f.write(f"  → エッチング反応は観測されませんでした\n")
                f.write("\n")

        print(f"レポートを保存しました: {report_path}")
        print("\n統計情報:")
        print(stats_df.to_string(index=False))


def main():
    """メイン実行関数"""
    print("=" * 70)
    print("Al2O3エッチング反応解析プログラム")
    print("=" * 70)
    print()

    # 解析オブジェクトの作成
    analyzer = Al2O3EtchingAnalyzer(data_dir="validation_etching")

    # データの読み込み
    analyzer.load_data()

    if not analyzer.data:
        print("エラー: 解析対象のデータが見つかりません")
        print("validation_etching ディレクトリに etching_test_*K_etching.log ファイルがあることを確認してください")
        return

    # 時系列プロットの作成
    print("時系列プロットを作成中...")
    analyzer.plot_time_series(save_fig=True)
    print()

    # 温度依存性のプロット
    print("温度依存性プロットを作成中...")
    analyzer.plot_temperature_dependence(save_fig=True)
    print()

    # レポートの生成
    print("解析レポートを生成中...")
    analyzer.generate_report()
    print()

    print("=" * 70)
    print("解析完了！")
    print("結果は analysis_results ディレクトリに保存されました")
    print("=" * 70)


if __name__ == "__main__":
    main()
