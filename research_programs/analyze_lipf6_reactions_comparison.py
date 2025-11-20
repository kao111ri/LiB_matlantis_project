#!/usr/bin/env python3
"""
LiPF₆反応シミュレーション統合解析プログラム

目的:
  - Phase 1-B (LiPF₆ + H₂O 加水分解)
  - Phase 1-C (LiPF₆ + Al 接触)
  - Phase 1-D (LiPF₆ + Al₂O₃ 接触)
  の3つのシミュレーション結果を統合的に解析・比較

解析内容:
  1. 各系でのHF生成の比較
  2. 温度依存性の比較
  3. 反応性の定量的評価
  4. Al vs Al₂O₃ の保護効果の比較
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class LiPF6ReactionComparison:
    """LiPF₆反応の統合解析クラス"""

    def __init__(self):
        """初期化"""
        self.data_dirs = {
            'hydrolysis': Path('phase1b_lipf6_hydrolysis_results'),
            'al': Path('phase1c_lipf6_al_results'),
            'al2o3': Path('phase1d_lipf6_al2o3_results'),
        }
        self.data: Dict[str, Dict[str, pd.DataFrame]] = {
            'hydrolysis': {},
            'al': {},
            'al2o3': {},
        }
        self.temperatures: List[float] = []

    def load_all_data(self) -> bool:
        """全データを読み込む"""
        print("=" * 70)
        print("データ読み込み開始")
        print("=" * 70)
        print()

        all_loaded = True

        # Phase 1-B: 加水分解
        print("【Phase 1-B: LiPF₆ + H₂O 加水分解】")
        if self.data_dirs['hydrolysis'].exists():
            pattern = self.data_dirs['hydrolysis'] / "*_reaction.log"
            files = sorted(glob.glob(str(pattern)))

            if files:
                for file in files:
                    fname = Path(file).stem.replace('_reaction', '')
                    # 温度を抽出
                    temp_str = fname.split('_')[-1].replace('K', '')
                    try:
                        temp = float(temp_str)
                        df = pd.read_csv(file)
                        self.data['hydrolysis'][f"{temp}K"] = df
                        if temp not in self.temperatures:
                            self.temperatures.append(temp)
                        print(f"  ✓ 読み込み: {Path(file).name} ({len(df)} steps)")
                    except:
                        print(f"  ✗ 読み込み失敗: {Path(file).name}")
            else:
                print(f"  ✗ データが見つかりません: {pattern}")
                all_loaded = False
        else:
            print(f"  ✗ ディレクトリが見つかりません: {self.data_dirs['hydrolysis']}")
            all_loaded = False
        print()

        # Phase 1-C: Al接触
        print("【Phase 1-C: LiPF₆ + Al 接触】")
        if self.data_dirs['al'].exists():
            pattern = self.data_dirs['al'] / "*_reaction.log"
            files = sorted(glob.glob(str(pattern)))

            if files:
                for file in files:
                    fname = Path(file).stem.replace('_reaction', '')
                    temp_str = fname.split('_')[-1].replace('K', '')
                    try:
                        temp = float(temp_str)
                        df = pd.read_csv(file)
                        self.data['al'][f"{temp}K"] = df
                        if temp not in self.temperatures:
                            self.temperatures.append(temp)
                        print(f"  ✓ 読み込み: {Path(file).name} ({len(df)} steps)")
                    except:
                        print(f"  ✗ 読み込み失敗: {Path(file).name}")
            else:
                print(f"  ✗ データが見つかりません: {pattern}")
                all_loaded = False
        else:
            print(f"  ✗ ディレクトリが見つかりません: {self.data_dirs['al']}")
            all_loaded = False
        print()

        # Phase 1-D: Al₂O₃接触
        print("【Phase 1-D: LiPF₆ + Al₂O₃ 接触】")
        if self.data_dirs['al2o3'].exists():
            pattern = self.data_dirs['al2o3'] / "*_reaction.log"
            files = sorted(glob.glob(str(pattern)))

            if files:
                for file in files:
                    fname = Path(file).stem.replace('_reaction', '')
                    temp_str = fname.split('_')[-1].replace('K', '')
                    try:
                        temp = float(temp_str)
                        df = pd.read_csv(file)
                        self.data['al2o3'][f"{temp}K"] = df
                        if temp not in self.temperatures:
                            self.temperatures.append(temp)
                        print(f"  ✓ 読み込み: {Path(file).name} ({len(df)} steps)")
                    except:
                        print(f"  ✗ 読み込み失敗: {Path(file).name}")
            else:
                print(f"  ✗ データが見つかりません: {pattern}")
                all_loaded = False
        else:
            print(f"  ✗ ディレクトリが見つかりません: {self.data_dirs['al2o3']}")
            all_loaded = False
        print()

        self.temperatures = sorted(list(set(self.temperatures)))

        print("=" * 70)
        print(f"データ読み込み完了: 温度条件 {len(self.temperatures)} 個")
        print(f"温度: {self.temperatures}")
        print("=" * 70)
        print()

        return all_loaded

    def plot_hf_comparison(self, save_fig: bool = True, output_dir: str = "comparison_results"):
        """HF生成の比較プロット"""
        if not any(self.data.values()):
            print("エラー: データが読み込まれていません")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        n_temps = len(self.temperatures)
        fig, axes = plt.subplots(1, n_temps, figsize=(6 * n_temps, 5))

        if n_temps == 1:
            axes = [axes]

        fig.suptitle('HF Formation Comparison: H₂O vs Al vs Al₂O₃',
                     fontsize=16, fontweight='bold')

        for idx, temp in enumerate(self.temperatures):
            ax = axes[idx]
            temp_label = f"{temp}K"

            # 加水分解
            if temp_label in self.data['hydrolysis']:
                df = self.data['hydrolysis'][temp_label]
                ax.plot(df['time[ps]'], df['n_HF'], 'o-', label='H₂O (hydrolysis)',
                       linewidth=2, markersize=4)

            # Al接触
            if temp_label in self.data['al']:
                df = self.data['al'][temp_label]
                ax.plot(df['time[ps]'], df['n_HF'], 's-', label='Al surface',
                       linewidth=2, markersize=4)

            # Al₂O₃接触
            if temp_label in self.data['al2o3']:
                df = self.data['al2o3'][temp_label]
                ax.plot(df['time[ps]'], df['n_HF'], '^-', label='Al₂O₃ surface',
                       linewidth=2, markersize=4)

            ax.set_xlabel('Time (ps)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of HF bonds', fontsize=12, fontweight='bold')
            ax.set_title(f'{temp_label}', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_fig:
            fig_path = output_path / "hf_comparison.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"✓ HF比較図を保存: {fig_path}")

        plt.show()

    def plot_surface_reaction_comparison(self, save_fig: bool = True,
                                        output_dir: str = "comparison_results"):
        """表面反応の比較（Al vs Al₂O₃）"""
        if not self.data['al'] and not self.data['al2o3']:
            print("警告: Al または Al₂O₃ のデータがありません")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        n_temps = len(self.temperatures)
        fig, axes = plt.subplots(2, n_temps, figsize=(6 * n_temps, 10))

        if n_temps == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle('Surface Corrosion Comparison: Al vs Al₂O₃',
                     fontsize=16, fontweight='bold')

        for idx, temp in enumerate(self.temperatures):
            temp_label = f"{temp}K"

            # Al-F結合（腐食）
            ax_alf = axes[0, idx]

            if temp_label in self.data['al']:
                df = self.data['al'][temp_label]
                ax_alf.plot(df['time[ps]'], df['n_AlF'], 'o-', label='Al surface',
                           linewidth=2, markersize=4, color='#e74c3c')

            if temp_label in self.data['al2o3']:
                df = self.data['al2o3'][temp_label]
                ax_alf.plot(df['time[ps]'], df['n_AlF'], 's-', label='Al₂O₃ surface',
                           linewidth=2, markersize=4, color='#3498db')

            ax_alf.set_xlabel('Time (ps)', fontsize=10)
            ax_alf.set_ylabel('Number of Al-F bonds', fontsize=10, fontweight='bold')
            ax_alf.set_title(f'Al-F Formation at {temp_label}', fontsize=12, fontweight='bold')
            ax_alf.legend(loc='best')
            ax_alf.grid(True, alpha=0.3)

            # P-F結合（分解）
            ax_pf = axes[1, idx]

            if temp_label in self.data['al']:
                df = self.data['al'][temp_label]
                ax_pf.plot(df['time[ps]'], df['n_PF'], 'o-', label='Al surface',
                          linewidth=2, markersize=4, color='#e74c3c')

            if temp_label in self.data['al2o3']:
                df = self.data['al2o3'][temp_label]
                ax_pf.plot(df['time[ps]'], df['n_PF'], 's-', label='Al₂O₃ surface',
                          linewidth=2, markersize=4, color='#3498db')

            ax_pf.set_xlabel('Time (ps)', fontsize=10)
            ax_pf.set_ylabel('Number of P-F bonds', fontsize=10, fontweight='bold')
            ax_pf.set_title(f'LiPF₆ Decomposition at {temp_label}', fontsize=12, fontweight='bold')
            ax_pf.legend(loc='best')
            ax_pf.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_fig:
            fig_path = output_path / "surface_reaction_comparison.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"✓ 表面反応比較図を保存: {fig_path}")

        plt.show()

    def calculate_reaction_rates(self) -> pd.DataFrame:
        """反応速度を計算"""
        results = []

        for system_type, system_data in self.data.items():
            for temp_label, df in system_data.items():
                temp = float(temp_label.replace('K', ''))

                # 最終時刻
                final_time = df['time[ps]'].iloc[-1]

                # HF生成速度（最終値/時間）
                final_hf = df['n_HF'].iloc[-1] if 'n_HF' in df.columns else 0
                hf_rate = final_hf / final_time if final_time > 0 else 0

                # 最大HF
                max_hf = df['n_HF'].max() if 'n_HF' in df.columns else 0

                # AlF生成（表面系のみ）
                final_alf = df['n_AlF'].iloc[-1] if 'n_AlF' in df.columns else 0
                max_alf = df['n_AlF'].max() if 'n_AlF' in df.columns else 0

                # P-F結合の変化（分解の指標）
                initial_pf = df['n_PF'].iloc[0] if 'n_PF' in df.columns else 0
                final_pf = df['n_PF'].iloc[-1] if 'n_PF' in df.columns else 0
                pf_change = initial_pf - final_pf

                results.append({
                    'System': system_type,
                    'Temperature_K': temp,
                    'HF_final': final_hf,
                    'HF_max': max_hf,
                    'HF_rate_per_ps': hf_rate,
                    'AlF_final': final_alf,
                    'AlF_max': max_alf,
                    'PF_decomposition': pf_change,
                    'Simulation_time_ps': final_time,
                })

        return pd.DataFrame(results)

    def generate_comparison_report(self, output_dir: str = "comparison_results"):
        """比較レポートを生成"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 反応速度の計算
        df_rates = self.calculate_reaction_rates()

        # CSVとして保存
        csv_path = output_path / "reaction_rates_comparison.csv"
        df_rates.to_csv(csv_path, index=False)
        print(f"✓ 反応速度データを保存: {csv_path}")

        # テキストレポート
        report_path = output_path / "comparison_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LiPF₆反応シミュレーション統合解析レポート\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"解析日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"比較対象:\n")
            f.write(f"  - Phase 1-B: LiPF₆ + H₂O (加水分解)\n")
            f.write(f"  - Phase 1-C: LiPF₆ + Al (表面接触)\n")
            f.write(f"  - Phase 1-D: LiPF₆ + Al₂O₃ (酸化膜接触)\n\n")

            f.write("-" * 80 + "\n")
            f.write("反応速度の比較\n")
            f.write("-" * 80 + "\n\n")
            f.write(df_rates.to_string(index=False))
            f.write("\n\n")

            f.write("-" * 80 + "\n")
            f.write("考察\n")
            f.write("-" * 80 + "\n\n")

            # システムごとの考察
            for system_type in ['hydrolysis', 'al', 'al2o3']:
                system_name = {
                    'hydrolysis': 'LiPF₆ + H₂O (加水分解)',
                    'al': 'LiPF₆ + Al (金属表面)',
                    'al2o3': 'LiPF₆ + Al₂O₃ (酸化膜)',
                }[system_type]

                f.write(f"【{system_name}】\n")

                system_df = df_rates[df_rates['System'] == system_type]

                if len(system_df) > 0:
                    avg_hf = system_df['HF_final'].mean()
                    avg_rate = system_df['HF_rate_per_ps'].mean()

                    f.write(f"  HF生成（平均）: {avg_hf:.2f}\n")
                    f.write(f"  HF生成速度（平均）: {avg_rate:.4f} bonds/ps\n")

                    if 'AlF_final' in system_df.columns and system_df['AlF_final'].sum() > 0:
                        avg_alf = system_df['AlF_final'].mean()
                        f.write(f"  Al-F結合（平均）: {avg_alf:.2f}\n")

                    # 温度依存性
                    if len(system_df) > 1:
                        temp_corr = system_df[['Temperature_K', 'HF_final']].corr().iloc[0, 1]
                        f.write(f"  温度とHF生成の相関: {temp_corr:.3f}\n")
                else:
                    f.write(f"  データがありません\n")

                f.write("\n")

            # Al vs Al₂O₃ の比較
            f.write("-" * 80 + "\n")
            f.write("Al vs Al₂O₃ 表面の比較\n")
            f.write("-" * 80 + "\n\n")

            al_df = df_rates[df_rates['System'] == 'al']
            al2o3_df = df_rates[df_rates['System'] == 'al2o3']

            if len(al_df) > 0 and len(al2o3_df) > 0:
                al_avg_alf = al_df['AlF_final'].mean()
                al2o3_avg_alf = al2o3_df['AlF_final'].mean()

                f.write(f"Al表面の平均Al-F結合: {al_avg_alf:.2f}\n")
                f.write(f"Al₂O₃表面の平均Al-F結合: {al2o3_avg_alf:.2f}\n\n")

                if al2o3_avg_alf < al_avg_alf:
                    protection_factor = (1 - al2o3_avg_alf / al_avg_alf) * 100 if al_avg_alf > 0 else 0
                    f.write(f"→ Al₂O₃酸化膜は約 {protection_factor:.1f}% の保護効果を示しています\n")
                else:
                    f.write(f"→ Al₂O₃表面でもAl表面と同等以上の反応が観測されました\n")
            else:
                f.write("比較に十分なデータがありません\n")

            f.write("\n")

            f.write("-" * 80 + "\n")
            f.write("結論\n")
            f.write("-" * 80 + "\n\n")
            f.write("1. HF生成反応の比較結果から、LiPF₆の分解反応性を評価\n")
            f.write("2. Al表面とAl₂O₃表面の反応性の違いから、酸化膜の保護効果を評価\n")
            f.write("3. 温度依存性から、実際のバッテリー動作温度での反応性を推定\n\n")

        print(f"✓ 比較レポートを保存: {report_path}")

        # レポート内容を表示
        print("\n" + "=" * 80)
        print("反応速度の比較")
        print("=" * 80)
        print(df_rates.to_string(index=False))
        print()


def main():
    """メイン実行関数"""
    print("\n" + "=" * 80)
    print("LiPF₆反応シミュレーション統合解析")
    print("=" * 80)
    print()

    # 解析オブジェクトの作成
    analyzer = LiPF6ReactionComparison()

    # データの読み込み
    if not analyzer.load_all_data():
        print("\n警告: 一部のデータが読み込めませんでした")
        print("利用可能なデータのみで解析を続行します\n")

    if not any(analyzer.data.values()):
        print("\nエラー: 解析対象のデータが見つかりません")
        print("以下のディレクトリにシミュレーション結果があることを確認してください:")
        for name, path in analyzer.data_dirs.items():
            print(f"  - {path}")
        return

    # HF生成の比較
    print("\n" + "=" * 80)
    print("HF生成の比較プロット作成中...")
    print("=" * 80)
    analyzer.plot_hf_comparison(save_fig=True)
    print()

    # 表面反応の比較
    print("=" * 80)
    print("表面反応の比較プロット作成中...")
    print("=" * 80)
    analyzer.plot_surface_reaction_comparison(save_fig=True)
    print()

    # レポート生成
    print("=" * 80)
    print("比較レポート生成中...")
    print("=" * 80)
    analyzer.generate_comparison_report()
    print()

    print("=" * 80)
    print("解析完了！")
    print("結果は comparison_results ディレクトリに保存されました")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
