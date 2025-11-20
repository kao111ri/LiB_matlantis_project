#!/usr/bin/env python3
"""
LiPF6.ipynbで行った加水分解反応シミュレーションの解析プログラム

解析内容:
- HF生成の時間変化
- LiF生成の時間変化
- PO結合の時間変化
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


class LiPF6HydrolysisAnalyzer:
    """LiPF6加水分解反応の解析クラス"""

    def __init__(self, data_dir: str = "step3_validation_md_cif"):
        """
        Parameters
        ----------
        data_dir : str
            解析対象のデータディレクトリ
        """
        self.data_dir = Path(data_dir)
        self.data: Dict[str, pd.DataFrame] = {}
        self.temperatures: List[float] = []

    def load_data(self, pattern: str = "md_*K_LiPF6_crystal_H2O_reaction.log") -> None:
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
            # ファイル名から温度を抽出 (例: md_350K_LiPF6_crystal_H2O_reaction.log -> 350)
            temp_str = log_file.stem.split('_')[1].replace('K', '')
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
        fig.suptitle('LiPF6 Hydrolysis Reaction Analysis', fontsize=16, fontweight='bold')

        colors = plt.cm.plasma(np.linspace(0, 1, len(self.data)))

        for idx, (temp_label, df) in enumerate(self.data.items()):
            color = colors[idx]

            # HF生成
            axes[0].plot(df['time[ps]'], df['n_HF'],
                        label=temp_label, color=color, linewidth=2, marker='o', markersize=4)

            # LiF生成
            axes[1].plot(df['time[ps]'], df['n_LiF'],
                        label=temp_label, color=color, linewidth=2, marker='s', markersize=4)

            # PO結合
            axes[2].plot(df['time[ps]'], df['n_PO'],
                        label=temp_label, color=color, linewidth=2, marker='^', markersize=4)

        # 軸ラベルと凡例の設定
        axes[0].set_ylabel('HF Bonds', fontsize=12, fontweight='bold')
        axes[0].set_title('HF Formation (Hydrolysis Product)', fontsize=12)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)

        axes[1].set_ylabel('LiF Bonds', fontsize=12, fontweight='bold')
        axes[1].set_title('LiF Formation', fontsize=12)
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)

        axes[2].set_xlabel('Time (ps)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('PO Bonds', fontsize=12, fontweight='bold')
        axes[2].set_title('P=O Double Bond Formation', fontsize=12)
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_fig:
            fig_path = output_path / "lipf6_hydrolysis_time_series.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"図を保存しました: {fig_path}")

        plt.show()

    def plot_combined_reactions(self, save_fig: bool = True, output_dir: str = "analysis_results") -> None:
        """
        全反応を1つの図にまとめてプロット

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

        n_temps = len(self.data)
        fig, axes = plt.subplots(1, n_temps, figsize=(6 * n_temps, 5))

        if n_temps == 1:
            axes = [axes]  # 1つの温度の場合もリストにする

        fig.suptitle('LiPF6 Hydrolysis: All Reactions', fontsize=16, fontweight='bold')

        for idx, (temp_label, df) in enumerate(self.data.items()):
            ax = axes[idx]

            ax.plot(df['time[ps]'], df['n_HF'], 'o-', label='HF', linewidth=2, markersize=6)
            ax.plot(df['time[ps]'], df['n_LiF'], 's-', label='LiF', linewidth=2, markersize=6)
            ax.plot(df['time[ps]'], df['n_PO'], '^-', label='P=O', linewidth=2, markersize=6)

            ax.set_xlabel('Time (ps)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Bonds', fontsize=12, fontweight='bold')
            ax.set_title(f'{temp_label}', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_fig:
            fig_path = output_path / "lipf6_hydrolysis_combined.png"
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
                'HF_final': df['n_HF'].iloc[-1],
                'HF_max': df['n_HF'].max(),
                'HF_mean': df['n_HF'].mean(),
                'LiF_final': df['n_LiF'].iloc[-1],
                'LiF_max': df['n_LiF'].max(),
                'LiF_mean': df['n_LiF'].mean(),
                'PO_final': df['n_PO'].iloc[-1],
                'PO_max': df['n_PO'].max(),
                'PO_mean': df['n_PO'].mean(),
                'Simulation_time_ps': df['time[ps]'].iloc[-1]
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
        fig.suptitle('Temperature Dependence of Hydrolysis Reactions', fontsize=16, fontweight='bold')

        # HF生成の温度依存性
        axes[0].plot(temps, stats_df['HF_final'], 'o-', markersize=10, linewidth=2, label='Final', color='#e74c3c')
        axes[0].plot(temps, stats_df['HF_max'], 's--', markersize=8, linewidth=2, label='Max', color='#c0392b')
        axes[0].set_xlabel('Temperature (K)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('HF Bonds', fontsize=12, fontweight='bold')
        axes[0].set_title('HF Formation', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # LiF生成の温度依存性
        axes[1].plot(temps, stats_df['LiF_final'], 'o-', markersize=10, linewidth=2, label='Final', color='#3498db')
        axes[1].plot(temps, stats_df['LiF_max'], 's--', markersize=8, linewidth=2, label='Max', color='#2980b9')
        axes[1].set_xlabel('Temperature (K)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('LiF Bonds', fontsize=12, fontweight='bold')
        axes[1].set_title('LiF Formation', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # PO結合の温度依存性
        axes[2].plot(temps, stats_df['PO_final'], 'o-', markersize=10, linewidth=2, label='Final', color='#2ecc71')
        axes[2].plot(temps, stats_df['PO_max'], 's--', markersize=8, linewidth=2, label='Max', color='#27ae60')
        axes[2].set_xlabel('Temperature (K)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('P=O Bonds', fontsize=12, fontweight='bold')
        axes[2].set_title('P=O Formation', fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_fig:
            fig_path = output_path / "lipf6_hydrolysis_temperature_dependence.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"図を保存しました: {fig_path}")

        plt.show()

    def analyze_reaction_rate(self, save_fig: bool = True, output_dir: str = "analysis_results") -> None:
        """
        反応速度の解析（時間微分）

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

        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle('Reaction Rate Analysis (d/dt)', fontsize=16, fontweight='bold')

        colors = plt.cm.plasma(np.linspace(0, 1, len(self.data)))

        for idx, (temp_label, df) in enumerate(self.data.items()):
            color = colors[idx]
            time = df['time[ps]'].values

            # HF生成速度
            hf_rate = np.gradient(df['n_HF'].values, time)
            axes[0].plot(time, hf_rate, label=temp_label, color=color, linewidth=2)

            # LiF生成速度
            lif_rate = np.gradient(df['n_LiF'].values, time)
            axes[1].plot(time, lif_rate, label=temp_label, color=color, linewidth=2)

            # PO生成速度
            po_rate = np.gradient(df['n_PO'].values, time)
            axes[2].plot(time, po_rate, label=temp_label, color=color, linewidth=2)

        # 軸ラベルと凡例の設定
        axes[0].set_ylabel('d(HF)/dt (bonds/ps)', fontsize=12, fontweight='bold')
        axes[0].set_title('HF Formation Rate', fontsize=12)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

        axes[1].set_ylabel('d(LiF)/dt (bonds/ps)', fontsize=12, fontweight='bold')
        axes[1].set_title('LiF Formation Rate', fontsize=12)
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

        axes[2].set_xlabel('Time (ps)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('d(PO)/dt (bonds/ps)', fontsize=12, fontweight='bold')
        axes[2].set_title('P=O Formation Rate', fontsize=12)
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)

        plt.tight_layout()

        if save_fig:
            fig_path = output_path / "lipf6_hydrolysis_reaction_rates.png"
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
        csv_path = output_path / "lipf6_hydrolysis_statistics.csv"
        stats_df.to_csv(csv_path, index=False)
        print(f"統計情報を保存しました: {csv_path}")

        # テキストレポートの生成
        report_path = output_path / "lipf6_hydrolysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("LiPF6 加水分解反応解析レポート\n")
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
                hf_final = df['n_HF'].iloc[-1]
                lif_final = df['n_LiF'].iloc[-1]
                po_final = df['n_PO'].iloc[-1]

                f.write(f"【{temp_label}】\n")
                f.write(f"  - HF生成数（最終）: {hf_final}\n")
                f.write(f"  - LiF生成数（最終）: {lif_final}\n")
                f.write(f"  - PO結合数（最終）: {po_final}\n")

                if hf_final > 0:
                    f.write(f"  → 加水分解反応が進行しています\n")
                    f.write(f"     LiPF6 + H2O → HF + ...\n")
                else:
                    f.write(f"  → 加水分解反応は観測されませんでした\n")
                f.write("\n")

            f.write("-" * 70 + "\n")
            f.write("反応機構の推定\n")
            f.write("-" * 70 + "\n\n")
            f.write("LiPF6 + H2O → LiF + POF3 + HF\n")
            f.write("POF3 + H2O → POF2OH + HF\n")
            f.write("POF2OH + H2O → POFO(OH)2 + HF\n")
            f.write("...\n\n")

        print(f"レポートを保存しました: {report_path}")
        print("\n統計情報:")
        print(stats_df.to_string(index=False))


def main():
    """メイン実行関数"""
    print("=" * 70)
    print("LiPF6加水分解反応解析プログラム")
    print("=" * 70)
    print()

    # 解析オブジェクトの作成
    analyzer = LiPF6HydrolysisAnalyzer(data_dir="step3_validation_md_cif")

    # データの読み込み
    analyzer.load_data()

    if not analyzer.data:
        print("エラー: 解析対象のデータが見つかりません")
        print("step3_validation_md_cif ディレクトリに md_*K_LiPF6_crystal_H2O_reaction.log ファイルがあることを確認してください")
        return

    # 時系列プロットの作成
    print("時系列プロットを作成中...")
    analyzer.plot_time_series(save_fig=True)
    print()

    # 全反応の統合プロット
    print("統合プロットを作成中...")
    analyzer.plot_combined_reactions(save_fig=True)
    print()

    # 温度依存性のプロット
    print("温度依存性プロットを作成中...")
    analyzer.plot_temperature_dependence(save_fig=True)
    print()

    # 反応速度の解析
    print("反応速度解析を実行中...")
    analyzer.analyze_reaction_rate(save_fig=True)
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
