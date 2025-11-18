"""
ファイルI/O関連のユーティリティ関数

このモジュールには以下の機能が含まれます：
- trajファイルからcifファイルへの変換
- 小さいファイルの削除とログ記録
"""

import os
import csv
from datetime import datetime
from typing import List, Dict
from ase.io import read, write


def convert_traj_to_cif(
    traj_filepath: str,
    cif_filepath: str = None,
    delete_traj: bool = True
) -> bool:
    """
    trajectoryファイルの最終構造をcifファイルに変換

    Args:
        traj_filepath (str): 入力trajectoryファイルパス
        cif_filepath (str): 出力cifファイルパス（Noneの場合は自動生成）
        delete_traj (bool): 変換後にtrajファイルを削除するか

    Returns:
        bool: 成功したらTrue
    """
    if cif_filepath is None:
        basename = os.path.splitext(traj_filepath)[0]
        cif_filepath = f"{basename}.cif"

    try:
        print(f"⏳ 処理中: {os.path.basename(traj_filepath)}")
        final_structure = read(traj_filepath, index=-1)
        write(cif_filepath, final_structure, format='cif')

        if os.path.exists(cif_filepath) and os.path.getsize(cif_filepath) > 0:
            print(f"  -> ✅ CIF作成成功: {os.path.basename(cif_filepath)}")

            if delete_traj:
                os.remove(traj_filepath)
                print(f"  -> 🗑️ TRAJ削除: {os.path.basename(traj_filepath)}")

            return True
        else:
            print(f"  -> ❌ CIF作成失敗: {os.path.basename(cif_filepath)}")
            return False

    except Exception as e:
        print(f"  -> ❌ エラー: {e}")
        return False


def clean_small_traj_files(
    target_dir: str,
    size_threshold: int = 2048,
    log_csv_filename: str = "small_traj_files_log.csv"
) -> Dict[str, List[str]]:
    """
    指定ディレクトリ内の小さいtrajファイルを削除し、ログを記録

    Args:
        target_dir (str): ターゲットディレクトリ
        size_threshold (int): ファイルサイズの閾値（バイト単位）
        log_csv_filename (str): ログCSVファイル名

    Returns:
        dict: 処理結果の統計情報
    """
    small_files_to_log = []
    results = {
        'deleted': [],
        'errors': []
    }

    print(f"--- 小さいTRAJファイルのクリーンアップを開始 ---")
    print(f"📁 ターゲットフォルダ: {target_dir}\n")

    if not os.path.isdir(target_dir):
        print(f"❌ エラー: 指定されたフォルダが見つかりません: {target_dir}")
        return results

    for filename in sorted(os.listdir(target_dir)):
        if filename.endswith(".traj"):
            traj_filepath = os.path.join(target_dir, filename)

            try:
                file_size = os.path.getsize(traj_filepath)
                if file_size < size_threshold:
                    print(f"🗑️  小さいファイルを検出: {filename} ({file_size} B) -> 削除")
                    small_files_to_log.append({
                        "filename": filename,
                        "size_bytes": file_size,
                        "deleted_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    os.remove(traj_filepath)
                    results['deleted'].append(filename)

            except OSError as e:
                print(f"⚠️ エラー: {filename} ({e})")
                results['errors'].append(filename)

    # ログファイルの書き出し
    if small_files_to_log:
        log_filepath = os.path.join(target_dir, log_csv_filename)
        print(f"\n📝 ログをCSVに保存: {log_filepath}")
        try:
            with open(log_filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['filename', 'size_bytes', 'deleted_at']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(small_files_to_log)
            print("  -> ✅ CSV保存完了。")
        except IOError as e:
            print(f"  -> ❌ CSV保存エラー: {e}")

    print(f"\n--- クリーンアップ完了: {len(results['deleted'])} 件削除 ---")
    return results


def batch_convert_traj_to_cif(
    target_dir: str,
    delete_traj: bool = True,
    skip_existing: bool = True
) -> Dict[str, List[str]]:
    """
    ディレクトリ内のすべてのtrajファイルをcifに一括変換

    Args:
        target_dir (str): ターゲットディレクトリ
        delete_traj (bool): 変換後にtrajファイルを削除するか
        skip_existing (bool): 既存のcifファイルをスキップするか

    Returns:
        dict: 処理結果の統計情報
    """
    results = {
        'created': [],
        'skipped': [],
        'errors': []
    }

    print(f"--- TRAJ → CIF 一括変換を開始 ---")
    print(f"📁 ターゲットフォルダ: {target_dir}\n")

    if not os.path.isdir(target_dir):
        print(f"❌ エラー: 指定されたフォルダが見つかりません: {target_dir}")
        return results

    for filename in sorted(os.listdir(target_dir)):
        if filename.endswith(".traj"):
            traj_filepath = os.path.join(target_dir, filename)
            basename = os.path.splitext(filename)[0]
            cif_filepath = os.path.join(target_dir, f"{basename}.cif")

            # 既存のcifファイルをチェック
            if skip_existing and os.path.exists(cif_filepath):
                print(f"👍 CIFは既に存在: {os.path.basename(cif_filepath)}")
                results['skipped'].append(os.path.basename(cif_filepath))
                continue

            # 変換実行
            if convert_traj_to_cif(traj_filepath, cif_filepath, delete_traj):
                results['created'].append(os.path.basename(cif_filepath))
            else:
                results['errors'].append(filename)

    # サマリー表示
    print("\n--- 処理結果サマリー ---")
    print(f"✅ 新たに作成されたCIFファイル: {len(results['created'])} 件")
    print(f"👍 既に存在したためスキップ: {len(results['skipped'])} 件")
    print(f"❌ エラーが発生: {len(results['errors'])} 件")
    print("-" * 30)

    return results
