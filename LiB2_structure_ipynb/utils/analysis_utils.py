"""
解析関連のユーティリティ関数

このモジュールには以下の機能が含まれます：
- 分子の断片カウント
- 予期しない分子の検出
- 分子の時間変化解析
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Optional
from ase.io import read
from pfcc_extras.structure.molecule import get_mol_list


def count_fragms(l_atoms: List, ref_fragm: List[str]) -> pd.DataFrame:
    """
    複数のフレームから分子の断片をカウントしてDataFrameを返す

    Args:
        l_atoms (list): ASE Atomsオブジェクトのリスト
        ref_fragm (list): 参照する分子式のリスト

    Returns:
        pd.DataFrame: 各フレームの分子カウント結果
    """
    df_list = []
    for idx, atoms in enumerate(l_atoms):
        d_formula = dict(zip(ref_fragm, np.zeros(len(ref_fragm), dtype=int)))
        d_formula['time[ps]'] = idx * 0.1 * 500 / 1000  # ps
        d_formula['T[K]'] = atoms.get_temperature()

        # 分子のリストを取得
        l_fragm_atoms = get_mol_list(atoms)[0]
        d_count = Counter([x.get_chemical_formula() for x in l_fragm_atoms])

        # 参照分子式のカウントを更新
        for fragm in ref_fragm:
            d_formula[fragm] = d_count[fragm]

        df_list.append(pd.DataFrame(d_formula, index=[idx]))

    if not df_list:
        return pd.DataFrame()

    return pd.concat(df_list)


def find_unexpected_molecules(
    l_traj_path: List[str],
    atoms_ini,
    ref_fragm: List[str]
) -> pd.DataFrame:
    """
    複数のtrajectoryファイルを解析し、参照リストに含まれない分子を
    出現回数と共に集計する

    Args:
        l_traj_path (list): trajectoryファイルパスのリスト
        atoms_ini: 初期構造のASE Atomsオブジェクト
        ref_fragm (list): 予め定義された分子式のリスト

    Returns:
        pd.DataFrame: 解析対象外の分子の 'Formula' と 'Total Count'
    """
    all_molecules_counter = Counter()
    ref_fragm_set = set(ref_fragm)

    print("解析を開始します...")

    for ftraj_path in l_traj_path:
        print(f"  - ファイルを処理中: {ftraj_path}")
        try:
            l_atoms = [atoms_ini] + read(ftraj_path, index=':')
        except FileNotFoundError:
            print(f"    ⚠️  警告: ファイルが見つかりませんでした。スキップします: {ftraj_path}")
            continue

        for atoms in l_atoms:
            mols, _ = get_mol_list(atoms)
            formulas_in_frame = [mol.get_chemical_formula() for mol in mols]
            all_molecules_counter.update(formulas_in_frame)

    # 参照リストに含まれない分子のみを抽出
    unexpected_molecules = {
        formula: count
        for formula, count in all_molecules_counter.items()
        if formula not in ref_fragm_set
    }

    if not unexpected_molecules:
        return pd.DataFrame(columns=['Formula', 'Total Count'])

    df = pd.DataFrame(
        unexpected_molecules.items(),
        columns=['Formula', 'Total Count']
    ).sort_values(by='Total Count', ascending=False).reset_index(drop=True)

    return df


def analyze_molecular_evolution_and_save(
    traj_path: str,
    atoms_ini,
    output_csv_path: str,
    timestep: float = 25.0
) -> None:
    """
    trajectoryファイルを解析し、すべての分子の量の時間変化を記録してCSVに保存

    Args:
        traj_path (str): trajectoryファイルパス
        atoms_ini: 初期構造のASE Atomsオブジェクト
        output_csv_path (str): 結果を保存するCSVファイルのパス
        timestep (float): タイムステップ (fs)
    """
    print(f"  - ファイルを処理中: {traj_path}")

    time_series_data = []

    try:
        l_atoms = [atoms_ini] + read(traj_path, index=':')
    except FileNotFoundError:
        print(f"    ⚠️  警告: ファイルが見つかりませんでした。スキップします: {traj_path}")
        return
    except Exception as e:
        print(f"    ❌ エラー: ファイルの読み込み中に問題が発生しました: {e}")
        return

    for i, atoms in enumerate(l_atoms):
        mols, _ = get_mol_list(atoms)
        counts = Counter([mol.get_chemical_formula() for mol in mols])

        frame_data = {
            'frame': i,
            'time_ps': (i - 1) * timestep / 1000 if i > 0 else 0,
            'temperature_K': atoms.get_temperature()
        }
        frame_data.update(counts)
        time_series_data.append(frame_data)

    # DataFrameを作成
    df = pd.DataFrame(time_series_data)
    df.fillna(0, inplace=True)

    # 基本情報以外の列を整数型に変換
    mol_cols = [col for col in df.columns if col not in ['frame', 'time_ps', 'temperature_K']]
    for col in mol_cols:
        df[col] = df[col].astype(int)

    # 列の順序を整理
    sorted_mol_cols = sorted(mol_cols)
    final_columns = ['frame', 'time_ps', 'temperature_K'] + sorted_mol_cols
    df = df[final_columns]

    # CSV保存
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"    ✅ 解析完了。結果を {output_csv_path} に保存しました。")


def analyze_trajectory_batch(
    traj_paths: List[str],
    atoms_ini,
    output_prefix: str = "evolution",
    timestep: float = 25.0
) -> None:
    """
    複数のtrajectoryファイルをバッチ解析してCSV出力

    Args:
        traj_paths (list): trajectoryファイルパスのリスト
        atoms_ini: 初期構造のASE Atomsオブジェクト
        output_prefix (str): 出力CSVファイル名のプレフィックス
        timestep (float): タイムステップ (fs)
    """
    print("バッチ解析を開始します...")

    for ftraj_path in traj_paths:
        base_name = ftraj_path.split('/')[-1].replace('.traj', '')
        output_csv = f"{output_prefix}_{base_name}.csv"
        analyze_molecular_evolution_and_save(ftraj_path, atoms_ini, output_csv, timestep)

    print("\nすべての処理が完了しました。")
