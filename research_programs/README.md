# Al/PVDF界面研究プログラム集

このディレクトリには、Al/PVDF界面の剥離性向上メカニズムを解明するための研究プログラムが含まれています。

## 📋 研究背景

**中心仮説**: Al/PVDF界面のAlF₃化が剥離性向上の主因

**未解明の3大疑問**:
1. **HFの主供給源は？** → PVDF熱分解 vs LiPF₆加水分解
2. **H₂Oの役割は？** → 化学反応(HF生成) vs 物理膨潤
3. **Al₂O₃厚さの影響は？** → F貫通阻害メカニズム

---

## 🚀 プログラム一覧

### Phase 1: 緊急課題(即実行)

#### 1. Phase 1-A: 既存データの徹底解析

**ファイル**: `phase1a_analysis_existing_data.py`

**目的**:
- 既存のtrajectory (Al2O3-PVDF_md_1600K_fix_al.traj) を解析
- HF生成、AlF₃生成、F貫通深度を追跡

**使用方法**:
```bash
# 1. 設定を編集
# INPUT_TRAJ_PATH を実際のtrajectoryファイルパスに設定

# 2. 実行
python phase1a_analysis_existing_data.py
```

**出力**:
- `phase1a_analysis_results/analysis_results.csv` - 解析データ
- `phase1a_analysis_results/analysis_results.png` - グラフ
- `phase1a_analysis_results/temperature_evolution.png` - 温度推移

**確認事項**:
- ✅ HF生成開始時刻
- ✅ AlF₃生成位置(表面 vs 界面)
- ✅ F原子のAl₂O₃層貫通可否

---

#### 2. Phase 1-B: LiPF₆加水分解検証

**ファイル**: `phase1b_lipf6_hydrolysis.py`

**目的**:
- LiPF₆結晶とH₂Oの混合系を加熱
- Matlantisポテンシャルの妥当性検証
- 350K(温水相当)と800K(高温)での反応速度比較

**使用方法**:
```bash
# 1. LiPF₆結晶ファイルを準備 (CIF形式)
# CONFIG['lipf6_cif_path'] を設定

# 2. 実行 (Matlantis環境が必要)
python phase1b_lipf6_hydrolysis.py
```

**出力**:
- `phase1b_lipf6_hydrolysis_results/initial_structure.xyz` - 初期構造
- `phase1b_lipf6_hydrolysis_results/lipf6_h2o_350K.traj` - 350K trajectory
- `phase1b_lipf6_hydrolysis_results/lipf6_h2o_800K.traj` - 800K trajectory
- `phase1b_lipf6_hydrolysis_results/reaction_comparison.png` - 比較グラフ
- `phase1b_lipf6_hydrolysis_results/reaction_summary.csv` - サマリー

**確認事項**:
- ✅ 350K でのHF生成速度
- ✅ 副生成物(LiF, POF₃)の確認
- ✅ 800Kとの比較

---

### Phase 2: 機構解明

#### 3. Phase 2-A: Al₂O₃層厚依存性検証

**ファイル**: `phase2a_al2o3_thickness_dependence.py`

**目的**:
- Al₂O₃層の厚さ(層数)を変えたNMC/Al₂O₃/PVDF系を構築
- 段階的加熱MD(LiPF₆分解→PVDF分解)を実行
- AlF₃生成量とF貫通性の層厚依存性を評価

**使用方法**:
```bash
# 1. パラメータを確認・調整
# CONFIG['al2o3_layers'] = [1, 2, 3, 5]  # テストする層数

# 2. 実行 (Matlantis環境が必要)
python phase2a_al2o3_thickness_dependence.py
```

**出力**:
- `phase2a_al2o3_thickness_results/initial_al2o3_*layers.xyz` - 各層の初期構造
- `phase2a_al2o3_thickness_results/al2o3_*layers_stage1.traj` - Stage1 (LiPF₆分解)
- `phase2a_al2o3_thickness_results/al2o3_*layers_stage2.traj` - Stage2 (PVDF分解)
- `phase2a_al2o3_thickness_results/thickness_dependence.png` - 層厚依存性グラフ
- `phase2a_al2o3_thickness_results/thickness_dependence.csv` - データ

**確認事項**:
- ✅ 臨界層厚(剥離性が急落する厚さ)の特定
- ✅ F貫通の活性化障壁
- ✅ 予想: 3層(~1.5nm)以上で急減

---

#### 4. PVDFアモルファス作成

**ファイル**: `pvdf_amorphous_builder.py`

**目的**:
- ランダムに配置したPVDF鎖をNPT計算(高温圧縮→冷却)
- 実験値に近い密度(~1.78 g/cm³)の樹脂モデルを構築

**使用方法**:
```bash
# 1. PVDF分子鎖の構造ファイルを準備
# CONFIG['pvdf_chain_file'] = 'pvdfchain.gjf'

# 2. 実行 (Matlantis環境が必要)
python pvdf_amorphous_builder.py
```

**出力**:
- `pvdf_amorphous_results/pvdf_initial.xyz` - 初期構造
- `pvdf_amorphous_results/pvdf_npt_compression.traj` - 圧縮計算trajectory
- `pvdf_amorphous_results/pvdf_npt_cooling.traj` - 冷却計算trajectory
- `pvdf_amorphous_results/pvdf_final_amorphous.xyz` - 最終構造
- `pvdf_amorphous_results/density_evolution.png` - 密度推移グラフ

**確認事項**:
- ✅ 最終密度が目標値(~1.78 g/cm³)に近いか
- ✅ セル形状が大きく歪んでいないか
- ✅ 分子が適切に緩和されているか

---

## 🔧 必要な環境

### 基本環境
- Python 3.8+
- ASE (Atomic Simulation Environment)
- NumPy
- Pandas
- Matplotlib

### Matlantis環境 (MD計算に必要)
- matlantis_features
- pfp_api_client
- pfcc_extras (LiquidGenerator)

### インストール例
```bash
pip install ase numpy pandas matplotlib
# Matlantis環境はMatlantisプラットフォームで提供されます
```

---

## 📊 推奨する実行順序

### 最優先 (今日中)
1. **Phase 1-A** (既存データ解析) - 計算不要、すぐに結果が得られます

### 第1週
2. **Phase 1-B** (LiPF₆加水分解検証) - 1週間以内に完了
3. **PVDFアモルファス作成** - 並行して実行可能

### 第2週以降
4. **Phase 2-A** (Al₂O₃層厚依存性) - 複数の系を計算するため時間がかかります

---

## 📈 期待される成果

### Phase 1完了時
- ✅ LiPF₆→HF生成を定量確認
- ✅ Al₂O₃臨界厚を特定(±0.5nm)

### Phase 2完了時
- ✅ 温水処理の優位性を説明(化学 vs 物理の比率)
- ✅ 実験への定量的予測提供

### 論文化の閾値
- **必須データ**: Phase 1 + Phase 2-A
- **補強データ**: PVDFアモルファス作成

---

## ⚠️ 注意事項

1. **trajectoryファイルのパス**: 各スクリプトの `CONFIG` セクションで、入力ファイルのパスを環境に合わせて設定してください。

2. **計算時間**: MD計算は時間がかかります。特にPhase 2-Aは複数の系を計算するため、数日かかる可能性があります。

3. **メモリ使用量**: 大規模な系では大量のメモリを使用します。必要に応じて、分子数やステップ数を調整してください。

4. **結果の検証**: 計算結果は必ず可視化して確認してください。異常な値や構造が見られる場合は、パラメータを調整して再計算してください。

---

## 📝 カスタマイズ方法

各スクリプトの先頭にある `CONFIG` 辞書を編集することで、計算条件をカスタマイズできます:

```python
CONFIG = {
    'n_molecules': 30,        # 分子数
    'temperature': 350.0,     # 温度 (K)
    'simulation_time': 20.0,  # 時間 (ps)
    'timestep': 0.5,          # タイムステップ (fs)
    # ...
}
```

---

## 🆘 トラブルシューティング

### Q1: "Matlantis環境が利用できません" と表示される
**A**: Matlantisプラットフォーム上で実行してください。ローカル環境では初期構造の生成のみ可能です。

### Q2: メモリ不足エラーが発生する
**A**: `CONFIG` で分子数 (`n_molecules`) やステップ数 (`n_steps`) を減らしてください。

### Q3: 計算が途中で止まる
**A**: ログファイル (*.log) を確認して、エラーメッセージを確認してください。多くの場合、構造が不安定なことが原因です。

---

## 📚 参考資料

- [ASE Documentation](https://wiki.fysik.dtu.dk/ase/)
- [Matlantis Documentation](https://matlantis.com/docs/)
- 研究方針の詳細: `../研究方針.md` (もしあれば)

---

## 🤝 貢献

バグ報告や改善提案は、GitHubのIssueまたはPull Requestでお願いします。

---

## 📄 ライセンス

このプロジェクトは研究目的で作成されています。

---

**最終更新**: 2025-11-18

**作成者**: Claude (AI Assistant)

**プロジェクト**: Al/PVDF界面の剥離性向上メカニズム解明
