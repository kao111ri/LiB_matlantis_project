"""
utilsモジュールの使用例

このスクリプトは、utilsモジュールの各機能を使った例を示します。
"""

# =============================================================================
# 例1: MD関連 - MDシミュレーションの実行
# =============================================================================

def example_md_simulation():
    """MDシミュレーションの実行例"""
    from ase.io import read
    from utils import run_md_simulation

    # 構造を読み込む
    atoms = read("input_structure.xyz")

    # MDシミュレーションを実行
    results = run_md_simulation(
        atoms=atoms,
        integrator_type="NVT_Berendsen",
        temperature=600.0,
        n_steps=10000,
        timestep=1.0,
        traj_file="output_md.traj",
        model_version='v7.0.0',
        calc_mode='CRYSTAL_U0'
    )

    print("MDシミュレーション完了")


# =============================================================================
# 例2: 最適化関連 - Matlantis最適化
# =============================================================================

def example_optimization():
    """構造最適化の実行例"""
    from ase.io import read
    from utils import run_matlantis_optimization

    # 構造を読み込む
    atoms = read("initial_structure.xyz")

    # 最適化を実行
    optimized_atoms = run_matlantis_optimization(
        atoms=atoms,
        trajectory_path="optimization.traj",
        fmax=0.05,
        name="my_structure",
        model_version='v7.0.0'
    )

    if optimized_atoms:
        print("最適化成功")
    else:
        print("最適化失敗")


# =============================================================================
# 例3: 構造操作関連 - 界面構築と密度計算
# =============================================================================

def example_structure_operations():
    """構造操作の実行例"""
    from ase.io import read, write
    from utils import (
        build_interface,
        calculate_density,
        create_water_unit_cell,
        set_cell_with_vacuum
    )

    # 水分子単位セルを作成
    water_cell = create_water_unit_cell(target_density=1.0)

    # セルを複製
    water_box = water_cell.repeat((5, 5, 3))

    # 真空層を追加
    water_box_with_vacuum = set_cell_with_vacuum(water_box, vacuum=2.0)

    # 2つのスラブから界面を構築
    slab1 = read("slab1.xyz")
    slab2 = read("slab2.xyz")
    interface = build_interface(
        slab1,
        slab2,
        target_xy=(12.0, 12.0),
        separation=2.0
    )

    write("interface.xyz", interface)

    # 密度を計算
    density = calculate_density(
        atoms=water_box,
        num_molecules=75,
        molar_mass=18.015  # H2O
    )
    print(f"密度: {density:.3f} g/cm³")


# =============================================================================
# 例4: 解析関連 - 分子の時間変化解析
# =============================================================================

def example_analysis():
    """解析の実行例"""
    from ase.io import read
    from utils import (
        count_fragms,
        find_unexpected_molecules,
        analyze_trajectory_batch
    )

    # 初期構造を読み込む
    atoms_ini = read('initial.traj', index=-1)

    # トラジェクトリファイルのリスト
    traj_paths = [
        'output/md_600K.traj',
        'output/md_800K.traj',
        'output/md_1000K.traj',
    ]

    # 参照する分子式
    ref_fragm = ["C12H12F12", "CH2O", "CO", "H2O", "H2", "CH4"]

    # 予期しない分子を検出
    unexpected_df = find_unexpected_molecules(traj_paths, atoms_ini, ref_fragm)

    if not unexpected_df.empty:
        print("予期しない分子が見つかりました:")
        print(unexpected_df)
        unexpected_df.to_csv('unexpected_molecules.csv', index=False)

    # 分子の時間変化を解析（バッチ処理）
    analyze_trajectory_batch(
        traj_paths=traj_paths,
        atoms_ini=atoms_ini,
        output_prefix="evolution",
        timestep=25.0
    )

    print("解析完了")


# =============================================================================
# 例5: ファイルI/O関連 - trajからcifへの変換
# =============================================================================

def example_file_io():
    """ファイルI/O操作の実行例"""
    from utils import (
        convert_traj_to_cif,
        batch_convert_traj_to_cif,
        clean_small_traj_files
    )

    # 単一ファイルの変換
    convert_traj_to_cif(
        traj_filepath="output/structure.traj",
        delete_traj=True
    )

    # ディレクトリ内の全trajファイルを一括変換
    results = batch_convert_traj_to_cif(
        target_dir="output",
        delete_traj=True,
        skip_existing=True
    )

    print(f"変換完了: {len(results['created'])} 件")

    # 小さいtrajファイルをクリーンアップ
    clean_results = clean_small_traj_files(
        target_dir="output",
        size_threshold=2048,
        log_csv_filename="small_files_log.csv"
    )

    print(f"削除完了: {len(clean_results['deleted'])} 件")


# =============================================================================
# 例6: すべての機能を組み合わせた例 - 完全なワークフロー
# =============================================================================

def example_complete_workflow():
    """完全なワークフローの例"""
    from ase.io import read, write
    from utils import (
        create_water_unit_cell,
        run_matlantis_optimization,
        run_md_simulation,
        analyze_molecular_evolution_and_save,
        convert_traj_to_cif
    )

    print("=== 完全なワークフロー例 ===\n")

    # ステップ1: 初期構造を作成
    print("ステップ1: 水分子ボックスを作成")
    water_cell = create_water_unit_cell(target_density=1.0)
    water_box = water_cell.repeat((5, 5, 3))
    write("water_box.xyz", water_box)

    # ステップ2: 構造最適化
    print("\nステップ2: 構造を最適化")
    optimized_atoms = run_matlantis_optimization(
        atoms=water_box,
        trajectory_path="water_opt.traj",
        fmax=0.05
    )

    if optimized_atoms is None:
        print("最適化失敗")
        return

    # ステップ3: MDシミュレーション
    print("\nステップ3: MDシミュレーションを実行")
    md_results = run_md_simulation(
        atoms=optimized_atoms,
        integrator_type="NVT_Berendsen",
        temperature=300.0,
        n_steps=5000,
        traj_file="water_md.traj"
    )

    # ステップ4: 解析
    print("\nステップ4: 軌跡を解析")
    analyze_molecular_evolution_and_save(
        traj_path="water_md.traj",
        atoms_ini=optimized_atoms,
        output_csv_path="water_evolution.csv"
    )

    # ステップ5: 最終構造をcifに変換
    print("\nステップ5: 最終構造をCIFに変換")
    convert_traj_to_cif("water_md.traj", delete_traj=False)

    print("\n=== ワークフロー完了 ===")


# =============================================================================
# メイン実行部分
# =============================================================================

if __name__ == "__main__":
    print("utilsモジュールの使用例\n")
    print("利用可能な例:")
    print("1. example_md_simulation() - MDシミュレーション")
    print("2. example_optimization() - 構造最適化")
    print("3. example_structure_operations() - 構造操作")
    print("4. example_analysis() - 解析")
    print("5. example_file_io() - ファイルI/O")
    print("6. example_complete_workflow() - 完全なワークフロー")
    print("\n各関数を呼び出して実行してください。")

    # 例: 完全なワークフローを実行する場合
    # example_complete_workflow()
