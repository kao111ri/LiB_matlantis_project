# LiB2 Structure Utils モジュール

このutilsモジュールは、ipynbファイル間で共通して使用されるコードをモジュール化したものです。

## モジュール構成

```
utils/
├── __init__.py              # モジュール初期化
├── md_utils.py              # MD関連
├── optimization_utils.py    # 最適化関連
├── structure_utils.py       # 構造操作関連
├── analysis_utils.py        # 解析関連
├── io_utils.py              # ファイルI/O関連
└── README.md                # このファイル
```

## インストール

utilsモジュールは、プロジェクトのルートディレクトリに配置されています。
Jupyter notebookまたはPythonスクリプトから直接インポートできます。

```python
# すべての関数をインポート
from utils import *

# または、特定の関数のみインポート
from utils import run_md_simulation, run_matlantis_optimization
```

## 主要な機能

### 1. MD関連 (`md_utils.py`)

分子動力学シミュレーションに関する関数とクラス。

**主要な関数:**
- `select_integrator()` - Integratorを選択
- `run_md_simulation()` - MDシミュレーションを実行
- `run_constant_temp_md()` - 一定温度でMDを実行
- `PrintWriteLog` - MDログを記録するクラス

**使用例:**
```python
from ase.io import read
from utils import run_md_simulation

atoms = read("structure.xyz")

results = run_md_simulation(
    atoms=atoms,
    integrator_type="NVT_Berendsen",
    temperature=600.0,
    n_steps=10000,
    traj_file="md_output.traj"
)
```

### 2. 最適化関連 (`optimization_utils.py`)

Matlantis PFPを使った構造最適化。複数のオプティマイザー、最適化履歴の追跡、可視化機能を提供。

**主要な関数・クラス:**
- `MatlantisOptimizer` - 最適化エンジンクラス（FIRE, LBFGS, BFGS対応）
- `optimize_structure_with_pfp()` - 統合最適化関数（推奨）
- `analyze_optimization_trajectory()` - 最適化結果の解析と可視化
- `run_matlantis_optimization()` - 従来版（FireLBFGS統合）

**使用例1: 統合関数（推奨）**
```python
from utils import optimize_structure_with_pfp

# 最適化、保存、解析を一括実行
optimized_atoms, results = optimize_structure_with_pfp(
    atoms=atoms,
    output_dir="optimization_results",
    name="my_structure",
    optimizer='FIRE',
    fmax=0.05,
    steps=200,
    model_version='v7.0.0',
    calc_mode='CRYSTAL_U0',
    fix_bottom_layers=3.0,  # 下層固定（オプション）
)

print(f"収束: {results['optimization_info']['converged']}")
print(f"エネルギー変化: {results['optimization_info']['energy_change']:.4f} eV")
```

**使用例2: MatlantisOptimizerクラス**
```python
from utils import MatlantisOptimizer

# オプティマイザーの初期化
optimizer = MatlantisOptimizer(
    model_version='v7.0.0',
    calc_mode='CRYSTAL_U0',
    verbose=True
)

# FIRE最適化
optimized, info = optimizer.optimize(
    atoms=atoms,
    optimizer='FIRE',
    fmax=0.05,
    steps=200,
    trajectory_path="opt.traj",
)

# LBFGS最適化
optimized, info = optimizer.optimize(
    atoms=atoms,
    optimizer='LBFGS',
    fmax=0.01,
    steps=100,
)
```

**使用例3: 最適化結果の解析**
```python
from utils import analyze_optimization_trajectory

analysis = analyze_optimization_trajectory(
    trajectory_path="opt.traj",
    output_dir="analysis_results",
)

print(f"エネルギー変化: {analysis['energy_change']:.4f} eV")
print(f"最終最大力: {analysis['final_fmax']:.4f} eV/Å")
```

**主な機能:**
- ✅ 複数のオプティマイザー（FIRE, LBFGS, BFGS）
- ✅ 下層原子の自動固定（表面計算向け）
- ✅ 最適化履歴の自動追跡
- ✅ エネルギーと力の収束グラフ作成
- ✅ 詳細なログ出力
- ✅ エラーハンドリング

### 3. 構造操作関連 (`structure_utils.py`)

構造の操作、セル設定、密度計算など。

**主要な関数:**
- `build_interface()` - 2つのスラブから界面を構築
- `calculate_density()` - 密度を計算
- `calculate_cell_from_density()` - 密度からセルサイズを計算
- `set_cell_with_vacuum()` - 真空層を追加
- `create_water_unit_cell()` - 水分子単位セルを作成

**使用例:**
```python
from utils import build_interface, create_water_unit_cell

# 水分子ボックスを作成
water_cell = create_water_unit_cell(target_density=1.0)
water_box = water_cell.repeat((5, 5, 3))

# 界面を構築
interface = build_interface(slab1, slab2, target_xy=(12.0, 12.0))
```

### 4. 解析関連 (`analysis_utils.py`)

軌跡の解析、分子のカウントなど。

**主要な関数:**
- `count_fragms()` - 分子の断片をカウント
- `find_unexpected_molecules()` - 予期しない分子を検出
- `analyze_molecular_evolution_and_save()` - 分子の時間変化を解析
- `analyze_trajectory_batch()` - 複数の軌跡をバッチ解析

**使用例:**
```python
from utils import analyze_trajectory_batch

analyze_trajectory_batch(
    traj_paths=['md1.traj', 'md2.traj'],
    atoms_ini=initial_atoms,
    output_prefix="evolution"
)
```

### 5. ファイルI/O関連 (`io_utils.py`)

ファイル形式の変換、クリーンアップなど。

**主要な関数:**
- `convert_traj_to_cif()` - trajファイルをcifに変換
- `batch_convert_traj_to_cif()` - 一括変換
- `clean_small_traj_files()` - 小さいファイルを削除

**使用例:**
```python
from utils import batch_convert_traj_to_cif

results = batch_convert_traj_to_cif(
    target_dir="output",
    delete_traj=True
)
```

## 使用例

詳細な使用例は `example_usage_utils.py` を参照してください。

## 既存のノートブックの移行

既存のipynbファイルをutilsモジュールを使うように更新する手順:

### Before（古い方法）:
```python
# 各ノートブックで同じコードを繰り返し定義
def run_matlantis_optimization(atoms, trajectory_path, fmax=0.05):
    print(f"最適化開始...")
    matlantis_atoms = MatlantisAtoms(atoms)
    # ... 長いコード ...
    return optimized_atoms

# 使用
optimized = run_matlantis_optimization(atoms, "opt.traj")
```

### After（utilsモジュールを使用）:
```python
# utilsモジュールからインポート
from utils import run_matlantis_optimization

# 使用（コードは同じ）
optimized = run_matlantis_optimization(atoms, "opt.traj")
```

## メリット

1. **コードの重複削減**: 同じコードを複数のノートブックに書く必要がなくなる
2. **メンテナンス性向上**: 関数を修正する際、1箇所を修正すればすべてのノートブックに反映される
3. **可読性向上**: ノートブックがよりシンプルになり、本質的なロジックに集中できる
4. **テスト容易性**: 共通関数を単体でテストできる
5. **再利用性**: 他のプロジェクトでも簡単に再利用できる

## トラブルシューティング

### インポートエラー
```python
ModuleNotFoundError: No module named 'utils'
```

**解決方法**: ノートブックがutilsディレクトリと同じ階層にあることを確認してください。

### パスの問題

ノートブックがサブディレクトリにある場合:
```python
import sys
sys.path.append('..')  # 親ディレクトリをパスに追加
from utils import *
```

## 更新履歴

- v1.1.0 (2025-11-21): 最適化機能の大幅拡張
  - `MatlantisOptimizer`クラスを追加（FIRE, LBFGS, BFGS対応）
  - `optimize_structure_with_pfp()`統合関数を追加
  - `analyze_optimization_trajectory()`解析・可視化機能を追加
  - 下層原子固定機能を追加
  - 最適化履歴の自動追跡機能を追加
  - エネルギーと力の収束グラフ作成機能を追加

- v1.0.0 (2025-10-31): 初版リリース
  - MD、最適化、構造操作、解析、ファイルI/O関連の機能を実装

## ライセンス

このモジュールは、プロジェクト固有のユーティリティです。

## お問い合わせ

質問や問題がある場合は、プロジェクトの担当者に連絡してください。
