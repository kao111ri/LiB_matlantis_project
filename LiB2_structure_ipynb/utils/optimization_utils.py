"""
構造最適化関連のユーティリティ関数

このモジュールには以下の機能が含まれます：
- Matlantisを使った構造最適化
"""

from pathlib import Path
from ase.io import write

from matlantis_features.atoms import MatlantisAtoms
from matlantis_features.features.common.opt import FireLBFGSASEOptFeature
from matlantis_features.utils.calculators import pfp_estimator_fn
from pfp_api_client.pfp.estimator import EstimatorCalcMode


def run_matlantis_optimization(
    atoms,
    trajectory_path: str,
    fmax: float = 0.05,
    name: str = "structure",
    model_version: str = 'v7.0.0',
    calc_mode: str = EstimatorCalcMode.CRYSTAL_PLUS_D3,
    n_run: int = 5000,
    show_progress_bar: bool = True,
):
    """
    Matlantisを使って構造最適化を実行する

    Args:
        atoms: ASE Atomsオブジェクト
        trajectory_path (str): 最適化トラジェクトリの保存先
        fmax (float): 収束判定基準（最大力、eV/Å）
        name (str): 計算の名前（ログ用）
        model_version (str): PFPモデルバージョン
        calc_mode (str): PFP計算モード
        n_run (int): 最大反復回数
        show_progress_bar (bool): 進捗バー表示

    Returns:
        最適化されたASE Atomsオブジェクト、またはNone（エラー時）
    """
    print(f"  -> Matlantis最適化開始 ({name}, fmax = {fmax}) ...")

    matlantis_atoms = MatlantisAtoms(atoms)
    estimator_function = pfp_estimator_fn(
        model_version=model_version,
        calc_mode=calc_mode
    )
    position_optimizer = FireLBFGSASEOptFeature(
        estimator_fn=estimator_function,
        filter=False,
        trajectory=str(trajectory_path),
        n_run=n_run,
        fmax=fmax,
        show_progress_bar=show_progress_bar
    )

    try:
        result = position_optimizer(matlantis_atoms)
        optimized_atoms = result.atoms.ase_atoms
        final_energy = result.output.energy_log[-1]
        print(f"  -> ✔️ 最適化完了！ エネルギー: {final_energy:.3f} eV")

        # XYZ形式でも保存
        xyz_save_path = Path(trajectory_path).with_suffix('.xyz')
        print(f"  -> 💾 最適化後の構造を保存します: {xyz_save_path.name}")
        write(str(xyz_save_path), optimized_atoms)

        return optimized_atoms

    except Exception as e:
        print(f"  -> ❌ 最適化中にエラーが発生: {e}")
        return None
