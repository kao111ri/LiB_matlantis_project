#!/usr/bin/env python3
"""
Packmol を使ったH2O充填のデモスクリプト

このスクリプトは、packmol_utils.pyを使用して、
既存の構造（例: LiPF6結晶）に水分子を充填する方法を示します。

【新機能】
- custom_box_size: H2O充填セルのサイズを指定可能
- n_molecules: 充填する水分子数を直接指定可能
- host_atoms: 充填したい構造（Atomsオブジェクト）を変更可能
"""

import numpy as np
import sys
from pathlib import Path
from ase.io import read, write
from ase.build import bulk
from ase import Atoms

# プロジェクトのutilsをインポート
sys.path.append(str(Path(__file__).parent.parent / "LiB2_structure_ipynb"))
from utils.packmol_utils import fill_box_with_packmol, check_packmol_command


def demo1_basic_usage():
    """
    デモ1: 基本的な使い方
    LiPF6構造に水を充填する（デフォルト設定）
    """
    print("\n" + "=" * 70)
    print("【デモ1】基本的な使い方: LiPF6に水を充填")
    print("=" * 70 + "\n")

    # LiPF6構造の読み込み（パスは環境に応じて変更してください）
    lipf6_path = "/home/jovyan/Kaori/MD/input/LiPF6.cif"

    if not Path(lipf6_path).exists():
        print(f"✗ LiPF6ファイルが見つかりません: {lipf6_path}")
        print("  代わりにAl構造で実行します\n")
        # フォールバック: Al構造を使用
        host = bulk('Al', 'fcc', a=4.05).repeat((3, 3, 3))
    else:
        host = read(lipf6_path)

    print(f"ホスト構造: {host.get_chemical_formula()}")
    print(f"原子数: {len(host)}")
    print(f"セルサイズ: {host.cell.cellpar()[:3].round(2)}")

    # Packmolで水を充填
    solvated = fill_box_with_packmol(
        host_atoms=host,
        solvent_type='H2O',
        density_g_cm3=0.9,  # 緩めの密度
        tolerance=2.2,
        verbose=True
    )

    # 結果を保存
    output_path = Path("demo_results") / "demo1_basic_h2o.xyz"
    output_path.parent.mkdir(exist_ok=True)
    write(output_path, solvated)

    print(f"\n✓ 結果を保存しました: {output_path}")
    print(f"  最終組成: {solvated.get_chemical_formula()}")
    print(f"  総原子数: {len(solvated)}\n")


def demo2_custom_box_size():
    """
    デモ2: カスタムボックスサイズの指定
    小さな構造を大きなボックスに配置して水を充填
    """
    print("\n" + "=" * 70)
    print("【デモ2】カスタムボックスサイズの指定")
    print("=" * 70 + "\n")

    # 小さなAl構造を作成
    host = bulk('Al', 'fcc', a=4.05).repeat((2, 2, 2))

    print(f"ホスト構造: {host.get_chemical_formula()}")
    print(f"元のセルサイズ: {host.cell.cellpar()[:3].round(2)} Å")

    # 25x25x25Åの大きなボックスに拡張して水を充填
    custom_size = (25.0, 25.0, 25.0)
    print(f"カスタムボックスサイズ: {custom_size} Å")

    solvated = fill_box_with_packmol(
        host_atoms=host,
        solvent_type='H2O',
        custom_box_size=custom_size,  # ★ カスタムボックスサイズ
        density_g_cm3=1.0,
        tolerance=2.0,
        verbose=True
    )

    # 結果を保存
    output_path = Path("demo_results") / "demo2_custom_box.xyz"
    write(output_path, solvated)

    print(f"\n✓ 結果を保存しました: {output_path}")
    print(f"  最終セルサイズ: {solvated.cell.cellpar()[:3].round(2)}")
    print(f"  総原子数: {len(solvated)}\n")


def demo3_specify_n_molecules():
    """
    デモ3: 水分子数を直接指定
    密度ではなく、充填する水分子数を直接指定する
    """
    print("\n" + "=" * 70)
    print("【デモ3】水分子数を直接指定")
    print("=" * 70 + "\n")

    # Al構造を作成
    host = bulk('Al', 'fcc', a=4.05).repeat((3, 3, 2))

    print(f"ホスト構造: {host.get_chemical_formula()}")
    print(f"セルサイズ: {host.cell.cellpar()[:3].round(2)} Å")

    # 正確に150個のH2O分子を充填
    n_water = 150
    print(f"充填する水分子数: {n_water}")

    solvated = fill_box_with_packmol(
        host_atoms=host,
        solvent_type='H2O',
        n_molecules=n_water,  # ★ 分子数を直接指定
        tolerance=2.0,
        verbose=True
    )

    # 結果を保存
    output_path = Path("demo_results") / "demo3_n_molecules.xyz"
    write(output_path, solvated)

    print(f"\n✓ 結果を保存しました: {output_path}")
    print(f"  最終組成: {solvated.get_chemical_formula()}")

    # 実際に追加された水分子数を確認
    n_water_atoms = len(solvated) - len(host)
    actual_n_water = n_water_atoms // 3
    print(f"  追加された水分子数: {actual_n_water}\n")


def demo4_different_solvents():
    """
    デモ4: 異なる溶媒の使用
    H2O以外の溶媒（メタノール、エタノールなど）を充填
    """
    print("\n" + "=" * 70)
    print("【デモ4】異なる溶媒の使用（メタノール）")
    print("=" * 70 + "\n")

    # Al構造を作成
    host = bulk('Al', 'fcc', a=4.05).repeat((2, 2, 2))

    print(f"ホスト構造: {host.get_chemical_formula()}")
    print(f"セルサイズ: {host.cell.cellpar()[:3].round(2)} Å")

    # メタノールを充填
    solvated = fill_box_with_packmol(
        host_atoms=host,
        solvent_type='CH3OH',  # ★ メタノール
        density_g_cm3=0.8,
        tolerance=2.0,
        verbose=True
    )

    # 結果を保存
    output_path = Path("demo_results") / "demo4_methanol.xyz"
    write(output_path, solvated)

    print(f"\n✓ 結果を保存しました: {output_path}")
    print(f"  最終組成: {solvated.get_chemical_formula()}\n")


def demo5_combined_options():
    """
    デモ5: 複数オプションの組み合わせ
    カスタムボックス + 分子数指定 + 異なる溶媒
    """
    print("\n" + "=" * 70)
    print("【デモ5】複数オプションの組み合わせ")
    print("=" * 70 + "\n")

    # LiPF6構造の読み込み
    lipf6_path = "/home/jovyan/Kaori/MD/input/LiPF6.cif"

    if not Path(lipf6_path).exists():
        print(f"✗ LiPF6ファイルが見つかりません: {lipf6_path}")
        print("  代わりにAl構造で実行します\n")
        host = bulk('Al', 'fcc', a=4.05).repeat((2, 2, 2))
    else:
        host = read(lipf6_path)

    print(f"ホスト構造: {host.get_chemical_formula()}")
    print(f"元のセルサイズ: {host.cell.cellpar()[:3].round(2)} Å")

    # カスタムボックス + 分子数指定
    custom_size = (20.0, 20.0, 25.0)
    n_water = 200

    print(f"カスタムボックスサイズ: {custom_size} Å")
    print(f"充填する水分子数: {n_water}")

    solvated = fill_box_with_packmol(
        host_atoms=host,
        solvent_type='H2O',
        custom_box_size=custom_size,  # ★ カスタムボックス
        n_molecules=n_water,           # ★ 分子数指定
        tolerance=2.2,
        verbose=True
    )

    # 結果を保存
    output_path = Path("demo_results") / "demo5_combined.xyz"
    write(output_path, solvated)

    print(f"\n✓ 結果を保存しました: {output_path}")
    print(f"  最終セルサイズ: {solvated.cell.cellpar()[:3].round(2)}")
    print(f"  最終組成: {solvated.get_chemical_formula()}")
    print(f"  総原子数: {len(solvated)}\n")


def main():
    """メイン実行関数"""
    print("\n" + "=" * 70)
    print("  Packmol H2O充填デモスクリプト")
    print("  新機能: custom_box_size, n_molecules オプション")
    print("=" * 70)

    # Packmolの確認
    if not check_packmol_command():
        print("\n✗ エラー: Packmolが見つかりません")
        print("  Packmolをインストールしてください")
        print("  例: conda install -c conda-forge packmol\n")
        return

    print("✓ Packmolが利用可能です\n")

    # 出力ディレクトリの作成
    Path("demo_results").mkdir(exist_ok=True)

    # 各デモを実行
    try:
        demo1_basic_usage()
    except Exception as e:
        print(f"✗ デモ1エラー: {e}\n")

    try:
        demo2_custom_box_size()
    except Exception as e:
        print(f"✗ デモ2エラー: {e}\n")

    try:
        demo3_specify_n_molecules()
    except Exception as e:
        print(f"✗ デモ3エラー: {e}\n")

    try:
        demo4_different_solvents()
    except Exception as e:
        print(f"✗ デモ4エラー: {e}\n")

    try:
        demo5_combined_options()
    except Exception as e:
        print(f"✗ デモ5エラー: {e}\n")

    print("\n" + "=" * 70)
    print("  全デモ完了")
    print("  結果ファイル: demo_results/ ディレクトリ")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
