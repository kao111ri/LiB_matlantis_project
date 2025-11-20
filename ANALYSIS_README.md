# MDシミュレーション解析プログラム

Al2O3_HF.ipynbとLiPF6.ipynbで行ったMDシミュレーションの計算結果を解析するためのプログラム群です。

## 概要

このプロジェクトには3つの解析プログラムが含まれています:

1. **analyze_al2o3_hf.py** - Al2O3表面のHFエッチング反応の解析
2. **analyze_lipf6.py** - LiPF6の加水分解反応の解析
3. **analyze_all.py** - 上記2つを統合的に解析

## 必要なパッケージ

```bash
pip install numpy pandas matplotlib pathlib
```

## ファイル構成

```
.
├── analyze_al2o3_hf.py          # Al2O3エッチング解析スクリプト
├── analyze_lipf6.py             # LiPF6加水分解解析スクリプト
├── analyze_all.py               # 統合解析スクリプト
├── ANALYSIS_README.md           # このファイル
│
├── validation_etching/          # Al2O3_HF.ipynb の出力ディレクトリ
│   └── etching_test_*K_etching.log
│
├── step3_validation_md_cif/     # LiPF6.ipynb の出力ディレクトリ
│   └── md_*K_LiPF6_crystal_H2O_reaction.log
│
└── analysis_results/            # 解析結果の出力ディレクトリ（自動生成）
    ├── *.png                    # グラフ画像
    ├── *.csv                    # 統計データ
    └── *.txt                    # レポートファイル
```

## 使い方

### 1. Al2O3エッチング反応の解析

```bash
python analyze_al2o3_hf.py
```

**入力データ:**
- `validation_etching/etching_test_*K_etching.log`

**出力ファイル:**
- `analysis_results/al2o3_etching_time_series.png` - 時系列プロット
- `analysis_results/al2o3_etching_temperature_dependence.png` - 温度依存性
- `analysis_results/al2o3_etching_statistics.csv` - 統計データ
- `analysis_results/al2o3_etching_report.txt` - 解析レポート

**解析内容:**
- Al-F結合数の時間変化（エッチング進行の指標）
- O-H結合数の時間変化（表面水酸基化）
- H2O分子生成の追跡
- 温度依存性の評価

### 2. LiPF6加水分解反応の解析

```bash
python analyze_lipf6.py
```

**入力データ:**
- `step3_validation_md_cif/md_*K_LiPF6_crystal_H2O_reaction.log`

**出力ファイル:**
- `analysis_results/lipf6_hydrolysis_time_series.png` - 時系列プロット
- `analysis_results/lipf6_hydrolysis_combined.png` - 統合プロット
- `analysis_results/lipf6_hydrolysis_temperature_dependence.png` - 温度依存性
- `analysis_results/lipf6_hydrolysis_reaction_rates.png` - 反応速度解析
- `analysis_results/lipf6_hydrolysis_statistics.csv` - 統計データ
- `analysis_results/lipf6_hydrolysis_report.txt` - 解析レポート

**解析内容:**
- HF生成の時間変化
- LiF生成の時間変化
- PO結合の時間変化
- 反応速度解析（時間微分）
- 温度依存性の評価

### 3. 統合解析（推奨）

```bash
python analyze_all.py
```

上記2つの解析を一度に実行し、統合レポートを生成します。

**出力ファイル:**
- 上記1と2のすべてのファイル
- `analysis_results/unified_analysis_report.txt` - 統合レポート

### オプション引数

```bash
# カスタムディレクトリを指定
python analyze_all.py --al2o3-dir /path/to/al2o3/data \
                      --lipf6-dir /path/to/lipf6/data \
                      --output-dir /path/to/output

# Al2O3エッチングのみ解析
python analyze_all.py --al2o3-only

# LiPF6加水分解のみ解析
python analyze_all.py --lipf6-only

# ヘルプを表示
python analyze_all.py --help
```

## 解析内容の詳細

### Al2O3エッチング反応（Al2O3_HF.ipynb）

**シミュレーション内容:**
- Al2O3スラブ表面にHF分子を配置
- NVT-MD（複数温度）でエッチング反応を観測

**追跡する反応:**
```
HF + Al2O3 → AlF3 + H2O
```

**解析指標:**
- **Al-F結合数**: エッチングの進行度
- **O-H結合数**: 表面水酸基の形成
- **H2O分子数**: 反応生成物

### LiPF6加水分解反応（LiPF6.ipynb）

**シミュレーション内容:**
- LiPF6結晶とH2O分子の混合系
- NVT-MD（複数温度）で加水分解を観測

**追跡する反応:**
```
LiPF6 + H2O → LiF + POF3 + HF
POF3 + H2O → POF2OH + HF
...
```

**解析指標:**
- **HF生成数**: 加水分解の主要生成物
- **LiF生成数**: 塩の形成
- **PO結合数**: POF3のP=O二重結合

## プログラムの主要機能

### クラスベースの設計

```python
# Al2O3エッチング解析
analyzer = Al2O3EtchingAnalyzer(data_dir="validation_etching")
analyzer.load_data()
analyzer.plot_time_series()
analyzer.plot_temperature_dependence()
analyzer.generate_report()

# LiPF6加水分解解析
analyzer = LiPF6HydrolysisAnalyzer(data_dir="step3_validation_md_cif")
analyzer.load_data()
analyzer.plot_time_series()
analyzer.plot_combined_reactions()
analyzer.plot_temperature_dependence()
analyzer.analyze_reaction_rate()
analyzer.generate_report()
```

### 統計解析機能

各解析プログラムは以下の統計情報を計算します:
- 最終値（final）
- 最大値（max）
- 平均値（mean）
- シミュレーション時間

### 可視化機能

- 時系列プロット（matplotlib）
- 温度依存性グラフ
- 複数条件の比較
- 反応速度の時間微分

## トラブルシューティング

### データが見つからない場合

```
エラー: 解析対象のデータが見つかりません
```

**対処法:**
1. データディレクトリのパスを確認
2. ログファイルの命名規則を確認
3. `--al2o3-dir` または `--lipf6-dir` オプションで正しいパスを指定

### モジュールのインポートエラー

```
ImportError: No module named 'pandas'
```

**対処法:**
```bash
pip install pandas matplotlib numpy
```

### プロットが表示されない場合

プログラムは自動的に画像ファイルとして保存します。
`analysis_results/` ディレクトリ内のPNGファイルを確認してください。

## 出力例

### 統計データ（CSV）

| Temperature | AlF_final | AlF_max | AlF_mean | OH_final | OH_max | OH_mean | H2O_final | H2O_max | H2O_mean |
|-------------|-----------|---------|----------|----------|--------|---------|-----------|---------|----------|
| 350K        | 15        | 18      | 12.5     | 7        | 10     | 6.8     | 0         | 2       | 0.5      |
| 600K        | 35        | 40      | 28.3     | 15       | 18     | 13.2    | 5         | 8       | 4.2      |

### レポート（TXT）

各解析プログラムは以下を含むテキストレポートを生成します:
- 解析日時
- データソース
- 統計サマリー
- 温度ごとの考察
- 反応機構の推定

## カスタマイズ

### データディレクトリの変更

プログラム内の以下の行を編集:

```python
# analyze_al2o3_hf.py
analyzer = Al2O3EtchingAnalyzer(data_dir="your/custom/path")

# analyze_lipf6.py
analyzer = LiPF6HydrolysisAnalyzer(data_dir="your/custom/path")
```

### ファイル名パターンの変更

`load_data()` メソッドの `pattern` 引数を変更:

```python
analyzer.load_data(pattern="custom_pattern_*.log")
```

### グラフのカスタマイズ

`plot_*()` メソッド内の matplotlib パラメータを編集:
- 色（colors）
- マーカー（marker）
- 線の太さ（linewidth）
- フォントサイズ（fontsize）

## 参考情報

### 反応機構

**Al2O3エッチング:**
```
6HF + Al2O3 → 2AlF3 + 3H2O
```

**LiPF6加水分解:**
```
LiPF6 + H2O → LiF + POF3 + HF
POF3 + H2O → POF2OH + HF
POF2OH + H2O → POFO(OH)2 + HF
POFO(OH)2 + H2O → PO(OH)3 + HF
```

### リチウムイオン電池との関連

- **HF生成**: 電池劣化の主要因
- **Al2O3コーティング**: 正極材料の保護層
- **LiPF6**: 一般的な電解質塩
- **加水分解**: 水分混入による劣化機構

## ライセンスと引用

このプログラムは研究目的で作成されました。
使用する際は適切に引用してください。

## 更新履歴

- 2025-11-20: 初版作成
  - Al2O3エッチング解析機能
  - LiPF6加水分解解析機能
  - 統合解析機能

## お問い合わせ

問題や改善提案がある場合は、プロジェクトの管理者に連絡してください。
