# 基板上PVDF分解解析ノートブック

## 概要

`analysis_substrate_PVDF.ipynb`は、Al2O3、AlF3などの基板上に載っているPVDFの分解プロセスを包括的に解析するためのJupyterノートブックです。

## 主な機能

### 1. 基本的な反応追跡
- **HF生成の時間変化**: PVDFの脱HF反応を定量的に追跡
- **AlF₃生成の時間変化**: 基板とフッ素の反応を監視
- **F原子の基板層への貫通深度**: フッ素原子がどれだけ深く基板に侵入するかを測定
- **O-H結合の形成**: 水分子の生成を追跡

### 2. 分子種の同定
以下の分子種を自動的に検出・定量します：
- HF（フッ化水素）
- CF₄（四フッ化炭素）
- C₂F₄（テトラフルオロエチレン）
- CO（一酸化炭素）
- CO₂（二酸化炭素）
- H₂（水素ガス）
- H₂O（水）
- AlF₃（三フッ化アルミニウム）

### 3. 複数ファイルの比較
複数のトラジェクトリファイルを同時に解析し、基板の種類や条件による違いを可視化できます。

## 使用方法

### 1. 設定

ノートブックの「設定パラメータ」セクションで、解析したいトラジェクトリファイルのパスを指定します：

```python
TRAJ_PATHS = [
    "/home/jovyan/Kaori/MD/LiB_2/structure/MD_PVDF/heat/PVDF_on_AlF3_cell_repeat_1x1x2_P0.000_T300K_HT500K.traj",
    "/home/jovyan/Kaori/MD/LiB_2/structure/MD_PVDF/heat/PVDF_on_Al2O3_cell_repeat_2x2x1_with_H2O_d2_0_P0.000_T300K_HT1600K.traj",
]
```

### 2. 解析パラメータのカスタマイズ

必要に応じて、結合判定距離などのパラメータを調整できます：

```python
ANALYSIS_PARAMS = {
    'HF_cutoff': 1.0,        # HF結合の判定距離 (Å)
    'AlF_cutoff': 2.0,       # Al-F結合の判定距離 (Å)
    'OH_cutoff': 1.1,        # O-H結合の判定距離 (Å)
    'CF_cutoff': 1.6,        # C-F結合判定 (Å)
    'CC_cutoff': 1.8,        # C-C結合判定 (Å)
    'CO_cutoff': 1.5,        # C-O結合判定 (Å)
    'HH_cutoff': 1.0,        # H-H結合判定 (Å)
    'frame_interval': 10,    # 解析するフレーム間隔
}
```

### 3. 実行

ノートブックを上から順に実行するだけで、以下が自動的に生成されます：
- 各ファイルの詳細解析グラフ（PNG）
- 解析結果のCSVファイル
- 複数ファイルの比較グラフ
- サマリーテーブル

## 出力ファイル

すべての出力は `substrate_pvdf_analysis_results/` ディレクトリに保存されます：

```
substrate_pvdf_analysis_results/
├── PVDF_on_AlF3_*.png              # AlF3上PVDFの解析グラフ
├── PVDF_on_AlF3_*.csv              # AlF3上PVDFの数値データ
├── PVDF_on_Al2O3_*.png             # Al2O3上PVDFの解析グラフ
├── PVDF_on_Al2O3_*.csv             # Al2O3上PVDFの数値データ
├── comparison_all.png              # 全ファイルの比較グラフ
└── analysis_summary.csv            # サマリーテーブル
```

## 解析グラフの内容

各解析グラフには以下の9つのサブプロットが含まれます：

1. **HF Generation**: HF分子の生成数の時間変化
2. **AlF₃ Formation**: Al-F結合の数の時間変化
3. **F Penetration**: F原子の基板への貫通深度
4. **CF₄ Formation**: CF₄分子の生成（高温副反応の指標）
5. **CO/CO₂ Formation**: 一酸化炭素と二酸化炭素の生成
6. **H₂/H₂O Formation**: 水素ガスと水分子の生成
7. **Temperature**: シミュレーション温度の推移
8. **AlF₃ Molecules**: AlF₃分子の数
9. **Analysis Summary**: 解析結果のテキストサマリー

## 比較グラフ

複数のトラジェクトリを解析した場合、自動的に比較グラフが生成されます：
- HF生成の比較
- AlF₃生成の比較
- F貫通深度の比較
- 温度プロファイルの比較

## サマリーテーブル

`analysis_summary.csv`には以下の情報が含まれます：
- ファイル名
- 総フレーム数
- シミュレーション時間
- 最大温度
- HF生成開始時刻
- AlF₃生成開始時刻
- 最大F貫通深度
- 最終HF数
- 最終Al-F結合数

## 必要なライブラリ

- numpy
- pandas
- matplotlib
- ase (Atomic Simulation Environment)
- pfcc_extras（オプション、分子種の同定に使用）

## 注意事項

1. **フレーム間隔**: デフォルトでは10フレームごとに解析します。全フレームを解析する場合は`frame_interval: 1`に設定してください（計算時間が大幅に増加します）。

2. **基板の自動検出**: 基板層は原子のtagまたはZ座標で自動的に判定されます。必要に応じて`analyze_trajectory`関数内の判定ロジックを調整してください。

3. **時間スケール**: タイムステップは0.1 fs/stepと仮定しています。実際の設定に合わせて調整してください。

## トラブルシューティング

### ファイルが見つからない

```
✗ エラー: ファイルが見つかりません: /path/to/file.traj
```

→ `TRAJ_PATHS`で指定したパスが正しいか確認してください。

### pfcc_extrasのインポートエラー

```
警告: pfcc_extrasが利用できません。一部の機能が制限されます。
```

→ pfcc_extrasは必須ではありません。基本的な解析機能は利用できます。

### メモリエラー

大きなトラジェクトリファイルでメモリエラーが発生する場合：
- `frame_interval`を大きくする（例：50や100）
- 一度に解析するファイル数を減らす

## 応用例

### 基板の影響を調査

```python
TRAJ_PATHS = [
    "PVDF_on_Al2O3.traj",
    "PVDF_on_AlF3.traj",
    "PVDF_on_Al_metal.traj",
]
```

### 温度依存性の調査

```python
TRAJ_PATHS = [
    "PVDF_on_Al2O3_500K.traj",
    "PVDF_on_Al2O3_1000K.traj",
    "PVDF_on_Al2O3_1600K.traj",
]
```

### 水の影響を調査

```python
TRAJ_PATHS = [
    "PVDF_on_Al2O3_dry.traj",
    "PVDF_on_Al2O3_with_H2O.traj",
]
```

## 更新履歴

- 2025-01-20: 初版作成
  - AlF3とAl2O3上のPVDFに対応
  - 複数ファイルの比較機能を追加
  - 分子種の自動同定機能を追加

## 関連ファイル

- `phase1a_analysis_existing_data.ipynb`: より高度な解析機能を持つノートブック（過剰加速判定など）
- `pvdf_analysis.ipynb`: 基本的なPVDF分解解析ノートブック

## ライセンス

このノートブックはプロジェクト内での使用を想定しています。
