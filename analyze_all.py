#!/usr/bin/env python3
"""
Al2O3_HF.ipynbとLiPF6.ipynbの計算結果を統合的に解析するプログラム

このスクリプトは両方のシミュレーション結果を読み込み、
包括的な解析レポートと可視化を生成します。
"""

import os
import sys
from pathlib import Path
import argparse

# 解析モジュールのインポート
try:
    from analyze_al2o3_hf import Al2O3EtchingAnalyzer
    from analyze_lipf6 import LiPF6HydrolysisAnalyzer
except ImportError:
    print("エラー: analyze_al2o3_hf.py または analyze_lipf6.py が見つかりません")
    print("これらのファイルが同じディレクトリにあることを確認してください")
    sys.exit(1)


class UnifiedAnalyzer:
    """統合解析クラス"""

    def __init__(
        self,
        al2o3_dir: str = "validation_etching",
        lipf6_dir: str = "step3_validation_md_cif",
        output_dir: str = "analysis_results"
    ):
        """
        Parameters
        ----------
        al2o3_dir : str
            Al2O3エッチングデータのディレクトリ
        lipf6_dir : str
            LiPF6加水分解データのディレクトリ
        output_dir : str
            解析結果の出力ディレクトリ
        """
        self.al2o3_dir = al2o3_dir
        self.lipf6_dir = lipf6_dir
        self.output_dir = output_dir

        # 各解析器のインスタンスを作成
        self.al2o3_analyzer = Al2O3EtchingAnalyzer(data_dir=al2o3_dir)
        self.lipf6_analyzer = LiPF6HydrolysisAnalyzer(data_dir=lipf6_dir)

    def run_all_analyses(self) -> None:
        """全ての解析を実行"""
        print("=" * 80)
        print("統合解析プログラム")
        print("Al2O3エッチング反応 & LiPF6加水分解反応")
        print("=" * 80)
        print()

        # 出力ディレクトリの作成
        Path(self.output_dir).mkdir(exist_ok=True)

        # --- Al2O3エッチング反応の解析 ---
        print("\n" + "=" * 80)
        print("1. Al2O3エッチング反応の解析")
        print("=" * 80)

        self.al2o3_analyzer.load_data()

        if self.al2o3_analyzer.data:
            print("\n[1-1] 時系列プロットを作成中...")
            self.al2o3_analyzer.plot_time_series(save_fig=True, output_dir=self.output_dir)

            print("\n[1-2] 温度依存性プロットを作成中...")
            self.al2o3_analyzer.plot_temperature_dependence(save_fig=True, output_dir=self.output_dir)

            print("\n[1-3] レポートを生成中...")
            self.al2o3_analyzer.generate_report(output_dir=self.output_dir)
        else:
            print(f"警告: Al2O3エッチングのデータが見つかりません（{self.al2o3_dir}）")

        # --- LiPF6加水分解反応の解析 ---
        print("\n" + "=" * 80)
        print("2. LiPF6加水分解反応の解析")
        print("=" * 80)

        self.lipf6_analyzer.load_data()

        if self.lipf6_analyzer.data:
            print("\n[2-1] 時系列プロットを作成中...")
            self.lipf6_analyzer.plot_time_series(save_fig=True, output_dir=self.output_dir)

            print("\n[2-2] 統合プロットを作成中...")
            self.lipf6_analyzer.plot_combined_reactions(save_fig=True, output_dir=self.output_dir)

            print("\n[2-3] 温度依存性プロットを作成中...")
            self.lipf6_analyzer.plot_temperature_dependence(save_fig=True, output_dir=self.output_dir)

            print("\n[2-4] 反応速度解析を実行中...")
            self.lipf6_analyzer.analyze_reaction_rate(save_fig=True, output_dir=self.output_dir)

            print("\n[2-5] レポートを生成中...")
            self.lipf6_analyzer.generate_report(output_dir=self.output_dir)
        else:
            print(f"警告: LiPF6加水分解のデータが見つかりません（{self.lipf6_dir}）")

        # --- 統合レポートの生成 ---
        print("\n" + "=" * 80)
        print("3. 統合レポートの生成")
        print("=" * 80)
        self.generate_unified_report()

        print("\n" + "=" * 80)
        print("全ての解析が完了しました！")
        print(f"結果は {self.output_dir} ディレクトリに保存されました")
        print("=" * 80)

    def generate_unified_report(self) -> None:
        """統合レポートを生成"""
        import pandas as pd

        report_path = Path(self.output_dir) / "unified_analysis_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("統合解析レポート\n")
            f.write("Al2O3エッチング反応 & LiPF6加水分解反応\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"解析日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"出力ディレクトリ: {self.output_dir}\n\n")

            # Al2O3エッチング反応のサマリー
            f.write("-" * 80 + "\n")
            f.write("【1】Al2O3エッチング反応（Al2O3_HF.ipynb）\n")
            f.write("-" * 80 + "\n\n")

            if self.al2o3_analyzer.data:
                f.write(f"解析ファイル数: {len(self.al2o3_analyzer.data)}\n")
                f.write(f"温度条件: {', '.join([str(t)+'K' for t in self.al2o3_analyzer.temperatures])}\n\n")

                f.write("【解析内容】\n")
                f.write("  - Al-F結合数の時間変化（エッチング進行）\n")
                f.write("  - O-H結合数の時間変化（表面水酸基化）\n")
                f.write("  - H2O分子生成の追跡\n\n")

                stats = self.al2o3_analyzer.calculate_statistics()
                f.write("【統計サマリー】\n")
                for _, row in stats.iterrows():
                    f.write(f"  {row['Temperature']}:\n")
                    f.write(f"    Al-F結合（最終）: {row['AlF_final']}\n")
                    f.write(f"    O-H結合（最終）: {row['OH_final']}\n")
                    f.write(f"    H2O分子（最終）: {row['H2O_final']}\n")
                f.write("\n")

                f.write("【生成ファイル】\n")
                f.write("  - al2o3_etching_time_series.png\n")
                f.write("  - al2o3_etching_temperature_dependence.png\n")
                f.write("  - al2o3_etching_statistics.csv\n")
                f.write("  - al2o3_etching_report.txt\n\n")
            else:
                f.write("データなし\n\n")

            # LiPF6加水分解反応のサマリー
            f.write("-" * 80 + "\n")
            f.write("【2】LiPF6加水分解反応（LiPF6.ipynb）\n")
            f.write("-" * 80 + "\n\n")

            if self.lipf6_analyzer.data:
                f.write(f"解析ファイル数: {len(self.lipf6_analyzer.data)}\n")
                f.write(f"温度条件: {', '.join([str(t)+'K' for t in self.lipf6_analyzer.temperatures])}\n\n")

                f.write("【解析内容】\n")
                f.write("  - HF生成の時間変化\n")
                f.write("  - LiF生成の時間変化\n")
                f.write("  - PO結合の時間変化\n")
                f.write("  - 反応速度解析\n\n")

                stats = self.lipf6_analyzer.calculate_statistics()
                f.write("【統計サマリー】\n")
                for _, row in stats.iterrows():
                    f.write(f"  {row['Temperature']}:\n")
                    f.write(f"    HF生成（最終）: {row['HF_final']}\n")
                    f.write(f"    LiF生成（最終）: {row['LiF_final']}\n")
                    f.write(f"    PO結合（最終）: {row['PO_final']}\n")
                f.write("\n")

                f.write("【生成ファイル】\n")
                f.write("  - lipf6_hydrolysis_time_series.png\n")
                f.write("  - lipf6_hydrolysis_combined.png\n")
                f.write("  - lipf6_hydrolysis_temperature_dependence.png\n")
                f.write("  - lipf6_hydrolysis_reaction_rates.png\n")
                f.write("  - lipf6_hydrolysis_statistics.csv\n")
                f.write("  - lipf6_hydrolysis_report.txt\n\n")
            else:
                f.write("データなし\n\n")

            # 総合考察
            f.write("-" * 80 + "\n")
            f.write("【総合考察】\n")
            f.write("-" * 80 + "\n\n")

            f.write("1. Al2O3表面のHFエッチング:\n")
            f.write("   - HFがAl2O3表面と反応し、Al-F結合を形成\n")
            f.write("   - 表面の水酸基化（O-H結合）も同時進行\n")
            f.write("   - 温度が高いほど反応が促進される傾向\n\n")

            f.write("2. LiPF6の加水分解反応:\n")
            f.write("   - LiPF6 + H2O → HF + LiF + POF3\n")
            f.write("   - 高温条件でより顕著な加水分解が観測される\n")
            f.write("   - HF生成は電池の劣化メカニズムと関連\n\n")

            f.write("3. 実験的意義:\n")
            f.write("   - リチウムイオン電池の劣化機構の理解\n")
            f.write("   - 電解液の安定性評価\n")
            f.write("   - 表面保護層の設計指針\n\n")

        print(f"統合レポートを保存しました: {report_path}")


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description="Al2O3_HF.ipynbとLiPF6.ipynbの計算結果を統合的に解析"
    )
    parser.add_argument(
        "--al2o3-dir",
        type=str,
        default="validation_etching",
        help="Al2O3エッチングデータのディレクトリ（デフォルト: validation_etching）"
    )
    parser.add_argument(
        "--lipf6-dir",
        type=str,
        default="step3_validation_md_cif",
        help="LiPF6加水分解データのディレクトリ（デフォルト: step3_validation_md_cif）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results",
        help="解析結果の出力ディレクトリ（デフォルト: analysis_results）"
    )
    parser.add_argument(
        "--al2o3-only",
        action="store_true",
        help="Al2O3エッチング反応のみを解析"
    )
    parser.add_argument(
        "--lipf6-only",
        action="store_true",
        help="LiPF6加水分解反応のみを解析"
    )

    args = parser.parse_args()

    # 統合解析器の作成
    analyzer = UnifiedAnalyzer(
        al2o3_dir=args.al2o3_dir,
        lipf6_dir=args.lipf6_dir,
        output_dir=args.output_dir
    )

    # 個別解析モードの処理
    if args.al2o3_only:
        print("Al2O3エッチング反応のみを解析します")
        analyzer.al2o3_analyzer.load_data()
        if analyzer.al2o3_analyzer.data:
            analyzer.al2o3_analyzer.plot_time_series(save_fig=True, output_dir=args.output_dir)
            analyzer.al2o3_analyzer.plot_temperature_dependence(save_fig=True, output_dir=args.output_dir)
            analyzer.al2o3_analyzer.generate_report(output_dir=args.output_dir)
        return

    if args.lipf6_only:
        print("LiPF6加水分解反応のみを解析します")
        analyzer.lipf6_analyzer.load_data()
        if analyzer.lipf6_analyzer.data:
            analyzer.lipf6_analyzer.plot_time_series(save_fig=True, output_dir=args.output_dir)
            analyzer.lipf6_analyzer.plot_combined_reactions(save_fig=True, output_dir=args.output_dir)
            analyzer.lipf6_analyzer.plot_temperature_dependence(save_fig=True, output_dir=args.output_dir)
            analyzer.lipf6_analyzer.analyze_reaction_rate(save_fig=True, output_dir=args.output_dir)
            analyzer.lipf6_analyzer.generate_report(output_dir=args.output_dir)
        return

    # 全解析の実行（デフォルト）
    analyzer.run_all_analyses()


if __name__ == "__main__":
    main()
