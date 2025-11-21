# 溶媒和系作成ツール（Solvation Utils）

LiPF₆やPVDFなどの構造に水分子を充填して溶媒和系を作成するユーティリティとデモノートブック。

## 📁 ファイル構成

### ユーティリティモジュール
- **`LiB2_structure_ipynb/utils/solvation_utils.py`**
  - 汎用的な溶媒充填関数（ランダム配置）
  - LiPF₆/PVDF専用のラッパー関数
  - 見積もり計算ヘルパー
- **`LiB2_structure_ipynb/utils/packmol_utils.py`** ⭐ NEW
  - **Packmol**を使った高速・確実な溶媒充填
  - カスタムボックスサイズ指定機能
  - 分子数の直接指定機能
  - 斜交セルの自動処理

### デモノートブック / スクリプト
- **`demo_solvation_lipf6_water.ipynb`** - LiPF₆ + 水の系（ランダム配置）
- **`demo_solvation_pvdf_water.ipynb`** - PVDF + 水の系（ランダム配置）
- **`demo_packmol_water_filling.py`** ⭐ NEW - Packmolを使ったH2O充填デモ

## 🚀 使い方

### 方法1: Packmol を使った充填 ⭐ 推奨

**メリット**: 高速・確実・カスタマイズ性が高い

```python
from ase.io import read, write
from utils.packmol_utils import fill_box_with_packmol

# LiPF6構造を読み込み
lipf6 = read('/home/jovyan/Kaori/MD/input/LiPF6.cif')

# 基本的な使い方: 密度指定
solvated = fill_box_with_packmol(
    host_atoms=lipf6,
    solvent_type='H2O',
    density_g_cm3=0.9,
    tolerance=2.2,
    verbose=True
)

# カスタムボックスサイズを指定（例: 25x25x30Å）
solvated = fill_box_with_packmol(
    host_atoms=lipf6,
    solvent_type='H2O',
    custom_box_size=(25.0, 25.0, 30.0),  # ★ 新機能
    density_g_cm3=1.0,
    tolerance=2.0,
    verbose=True
)

# 分子数を直接指定
solvated = fill_box_with_packmol(
    host_atoms=lipf6,
    solvent_type='H2O',
    n_molecules=200,  # ★ 新機能
    tolerance=2.2,
    verbose=True
)

# 保存
write('lipf6_water.xyz', solvated)
```

**必須**: `packmol` コマンドがインストールされている必要があります
```bash
conda install -c conda-forge packmol
```

### 方法2: ランダム配置を使った充填

**メリット**: 外部依存なし、細かい制御が可能

```python
from ase.io import read
from utils.solvation_utils import fill_lipf6_with_water

# LiPF6構造を読み込み
lipf6 = read('/home/jovyan/Kaori/MD/input/LiPF6.cif')

# 水を充填（密度0.9 g/cm³）
solvated = fill_lipf6_with_water(
    lipf6,
    water_density=0.9,
    min_distance=2.2,
    random_seed=42
)

# 保存
write('lipf6_water.xyz', solvated)
```

### 汎用的な使用例（他の溶媒）

```python
from utils.solvation_utils import fill_box_with_molecules

# メタノールで充填
solvated = fill_box_with_molecules(
    host_atoms=structure,
    solvent_type='CH3OH',  # ASE moleculeで認識される名前
    density_g_cm3=0.79,
    min_distance=2.0
)
```

## 📊 主な機能

### 🆕 Packmol方式の新機能

#### 1. カスタムボックスサイズ指定
セルサイズを自由に設定して水を充填：
```python
solvated = fill_box_with_packmol(
    host_atoms=structure,
    solvent_type='H2O',
    custom_box_size=(30.0, 30.0, 40.0),  # x, y, z (Å)
    density_g_cm3=1.0
)
```

#### 2. 分子数の直接指定
密度ではなく、充填する分子数を直接指定：
```python
solvated = fill_box_with_packmol(
    host_atoms=structure,
    solvent_type='H2O',
    n_molecules=500  # 正確に500個のH2O
)
```

#### 3. 斜交セルの自動処理
非直交セルを自動的に直方体に変換してPackmolで処理

#### 4. 高速・確実な配置
Packmolの最適化アルゴリズムで原子の重なりを確実に回避

### 共通機能（両方式）

#### 1. 密度ベースの分子数計算
指定した密度（g/cm³）から必要な溶媒分子数を自動計算：

```python
estimate = estimate_required_molecules(atoms, 'H2O', 1.0)
print(f"必要な水分子数: {estimate['n_molecules_required']}")
```

#### 2. ランダム配置と重なりチェック（solvation_utils.py）
- ランダムな位置と回転で分子を配置
- 既存原子との距離チェック
- 周期境界条件（PBC）対応

### カスタマイズ可能なパラメータ

#### Packmol方式 (`packmol_utils.py`)

| パラメータ | 説明 | デフォルト |
|----------|------|----------|
| `host_atoms` | 充填対象の構造 | 必須 |
| `solvent_type` | 溶媒分子（H2O, CH3OH等） | 'H2O' |
| `density_g_cm3` | 溶媒の目標密度 | 1.0 |
| `tolerance` | 原子間最小距離 (Å) | 2.0 |
| `custom_box_size` | カスタムボックス (a, b, c) | None (自動) |
| `n_molecules` | 分子数を直接指定 | None (密度から計算) |
| `seed` | 乱数シード | -1 (ランダム) |

#### ランダム配置方式 (`solvation_utils.py`)

| パラメータ | 説明 | デフォルト |
|----------|------|----------|
| `density_g_cm3` | 溶媒の目標密度 | 1.0 |
| `min_distance` | 最小許容距離 (Å) | 2.0 |
| `target_fill_fraction` | 目標達成率 | 1.0 |
| `max_attempts_per_molecule` | 1分子あたりの試行回数 | 1000 |
| `random_seed` | 乱数シード（再現性） | None |

## 🔬 応用例

### 1. LiPF₆ + 水系
```python
# 低密度（緩めの初期配置）
solvated = fill_lipf6_with_water(lipf6, water_density=0.8)
# → NPT緩和で実密度（~1.0 g/cm³）に調整
```

### 2. PVDF + 水系（含水率の調整）
```python
# 低含水（わずかに湿潤）
pvdf_low = fill_pvdf_with_water(pvdf, water_density=0.3)

# 中含水（部分的に充填）
pvdf_mid = fill_pvdf_with_water(pvdf, water_density=0.7)

# 高含水（完全に充填）
pvdf_high = fill_pvdf_with_water(pvdf, water_density=1.0)
```

### 3. 他の溶媒（エタノール、メタノールなど）
```python
# エタノールで充填
ethanol_system = fill_box_with_molecules(
    structure,
    solvent_type='C2H5OH',
    density_g_cm3=0.789
)
```

## 📈 推奨設定

### LiPF₆系
```python
CONFIG = {
    'water_density_g_cm3': 0.9,      # 緩めの初期配置
    'min_distance': 2.2,              # LiPF6のFと水のH
    'target_fill_fraction': 0.85,    # 85%達成でOK
    'max_attempts_per_molecule': 2000
}
```

### PVDF系（密な構造）
```python
CONFIG = {
    'water_density_g_cm3': 0.85,     # より緩めに
    'min_distance': 2.2,              # PVDFのFと水のH
    'target_fill_fraction': 0.80,    # 80%達成でOK
    'max_attempts_per_molecule': 3000 # 試行回数を多めに
}
```

## ⚙️ パラメータ選択のガイドライン

### 密度設定
- **初期配置**: 0.8-0.9 g/cm³（緩め）
  - 配置が容易
  - 後のNPT緩和で調整
- **実密度**: 1.0 g/cm³（水の標準密度）
  - 配置が困難
  - 計算時間が長い

### 最小距離（min_distance）
| 原子ペア | 推奨値 (Å) |
|---------|-----------|
| F (LiPF6/PVDF) - H (H2O) | 2.0-2.5 |
| C/O - O (H2O) | 2.2-2.8 |
| 一般的なファンデルワールス | 2.0-2.5 |

### 目標達成率（target_fill_fraction）
- **疎な構造**: 0.9-1.0（ほぼ全て配置可能）
- **密な構造**: 0.7-0.8（PVDFなど）
- **非常に密**: 0.5-0.7（空隙が少ない場合）

## 🔄 ワークフロー例

### 完全なシミュレーション準備

```python
# 1. 構造読み込み
lipf6 = read('LiPF6.cif')

# 2. 水充填（緩めの初期配置）
solvated = fill_lipf6_with_water(lipf6, water_density=0.85)

# 3. 初期構造保存
write('lipf6_water_initial.xyz', solvated)

# 4. NPT緩和（Matlantis）
# ... NPT MD で密度を1.0 g/cm³ に調整 ...

# 5. NVT平衡化
# ... 300K で平衡化 ...

# 6. 反応シミュレーション
# ... 高温（600-800K）でLiPF₆の加水分解を観測 ...
```

## 📝 出力ファイル

デモノートブックを実行すると以下が生成されます：

```
solvation_results/
├── lipf6_h2o_d0_9_initial.xyz      # 初期構造（可視化用）
├── lipf6_h2o_d0_9_initial.cif      # 初期構造（結晶情報保持）
├── lipf6_h2o_d0_9_stats.txt        # 統計情報
└── pvdf_h2o_d0_85_distribution.png # 水分子分布図（PVDF）
```

## 🐛 トラブルシューティング

### 配置できる分子数が少ない
**原因**: 構造が密すぎる、または`min_distance`が大きすぎる

**解決策**:
1. `water_density_g_cm3`を下げる（0.7-0.8）
2. `min_distance`を下げる（2.0-2.2）
3. `target_fill_fraction`を下げる（0.7-0.8）
4. `max_attempts_per_molecule`を増やす（3000-5000）

### 計算時間が長い
**原因**: 目標分子数が多すぎる、または構造が密

**解決策**:
1. `target_fill_fraction`を下げる（0.8）
2. `water_density_g_cm3`を下げる
3. 並列化（将来的な拡張）

### 密度が目標に到達しない
**原因**: 物理的に配置不可能

**解決策**:
1. これは正常（実際の空隙量を反映）
2. NPT緩和で密度を調整
3. より小さいセルで試す

## 🔗 関連ファイル

- `phase1c_lipf6_al_contact.ipynb` - LiPF₆+基板のMDシミュレーション
- `reaxFF_H2O_PVDF_Al_shrink.ipynb` - PVDF+基板+H2Oの界面系

## 📚 参考資料

### 水の物性
- 密度: 1.0 g/cm³ (25°C)
- モル質量: 18.015 g/mol
- ファンデルワールス半径:
  - O: 1.52 Å
  - H: 1.20 Å

### LiPF₆の物性
- 密度（結晶）: ~2.8 g/cm³
- モル質量: 151.91 g/mol

### PVDFの物性
- 密度: ~1.78 g/cm³
- モノマー（CH₂CF₂）: 64.03 g/mol

## 💡 今後の拡張

- [x] ✅ **Packmol統合** - 高速・確実な充填（完了）
- [x] ✅ **カスタムボックスサイズ** - 柔軟なセルサイズ設定（完了）
- [x] ✅ **分子数直接指定** - 密度以外の指定方法（完了）
- [ ] 複数の溶媒種の同時配置
- [ ] グリッドベースの高速配置アルゴリズム
- [ ] イオン（Li⁺, PF₆⁻）の個別配置
- [ ] 界面系（基板/溶媒）の自動構築
- [ ] 濃度勾配の作成
- [ ] 並列化による高速化

## 🤝 貢献

改善提案やバグ報告は Issue でお願いします。

---

**作成日**: 2025-11-21
**更新日**: 2025-11-21 (Packmol機能追加)
**バージョン**: 1.1
