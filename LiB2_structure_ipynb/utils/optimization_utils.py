"""
構造最適化関連のユーティリティ関数

このモジュールには以下の機能が含まれます：
- Matlantis PFPを使った構造最適化
- 複数のオプティマイザー（FIRE, LBFGS, BFGS）のサポート
- 最適化履歴の追跡と可視化
- 柔軟な設定オプション
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple
from ase import Atoms
from ase.io import write, Trajectory
from ase.optimize import FIRE, LBFGS, BFGS
from ase.constraints import FixAtoms

# Matlantis関連のインポート
from matlantis_features.atoms import MatlantisAtoms
from matlantis_features.features.common.opt import FireLBFGSASEOptFeature
from matlantis_features.utils.calculators import pfp_estimator_fn
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator


# ========================================================================
# 既存の関数（互換性のため保持）
# ========================================================================

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
    Matlantisを使って構造最適化を実行する（FireLBFGS統合版）

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


# ========================================================================
# 新規：ASE Optimizer経由のMatlantis PFP最適化
# ========================================================================

class MatlantisOptimizer:
    """
    Matlantis PFPを使ったASE構造最適化のラッパークラス

    複数のオプティマイザー（FIRE, LBFGS, BFGS）をサポートし、
    最適化履歴の追跡と可視化機能を提供します。
    """

    SUPPORTED_OPTIMIZERS = {
        'FIRE': FIRE,
        'LBFGS': LBFGS,
        'BFGS': BFGS,
    }

    def __init__(
        self,
        model_version: str = 'v7.0.0',
        calc_mode: Union[str, EstimatorCalcMode] = EstimatorCalcMode.CRYSTAL_U0,
        verbose: bool = True,
    ):
        """
        Args:
            model_version: Matlantisモデルのバージョン
            calc_mode: 計算モード（CRYSTAL_U0, CRYSTAL_PLUS_D3等）
            verbose: 詳細ログの出力
        """
        self.model_version = model_version
        self.calc_mode = calc_mode
        self.verbose = verbose

        # Estimatorの初期化
        self.estimator = Estimator(
            calc_mode=calc_mode,
            model_version=model_version
        )

        # Calculatorの初期化
        self.calculator = ASECalculator(self.estimator)

        if self.verbose:
            print(f"✓ MatlantisOptimizer初期化完了")
            print(f"  モデルバージョン: {model_version}")
            print(f"  計算モード: {calc_mode}")

    def optimize(
        self,
        atoms: Atoms,
        optimizer: str = 'FIRE',
        fmax: float = 0.05,
        steps: int = 200,
        trajectory_path: Optional[str] = None,
        logfile: Optional[str] = None,
        fix_bottom_layers: Optional[float] = None,
        **optimizer_kwargs
    ) -> Tuple[Atoms, Dict]:
        """
        構造最適化を実行

        Args:
            atoms: 最適化するASE Atomsオブジェクト
            optimizer: 使用するオプティマイザー ('FIRE', 'LBFGS', 'BFGS')
            fmax: 収束判定基準（最大力、eV/Å）
            steps: 最大ステップ数
            trajectory_path: Trajectoryファイルの保存先（オプション）
            logfile: ログファイルのパス（オプション）
            fix_bottom_layers: 下層原子を固定する高さ閾値（Å、オプション）
            **optimizer_kwargs: オプティマイザーへの追加引数

        Returns:
            (optimized_atoms, optimization_info):
                - optimized_atoms: 最適化後のAtomsオブジェクト
                - optimization_info: 最適化情報の辞書
        """
        if optimizer not in self.SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"未対応のオプティマイザー: {optimizer}. "
                f"サポートされているのは: {list(self.SUPPORTED_OPTIMIZERS.keys())}"
            )

        # Atomsのコピーを作成
        atoms_opt = atoms.copy()

        # 下層原子の固定（オプション）
        if fix_bottom_layers is not None:
            atoms_opt = self._fix_bottom_atoms(atoms_opt, fix_bottom_layers)

        # Calculatorの設定
        atoms_opt.calc = self.calculator

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  構造最適化開始")
            print(f"{'='*60}")
            print(f"オプティマイザー: {optimizer}")
            print(f"収束基準 (fmax): {fmax} eV/Å")
            print(f"最大ステップ数: {steps}")
            print(f"原子数: {len(atoms_opt)}")
            if fix_bottom_layers is not None:
                n_fixed = len([c for c in atoms_opt.constraints if isinstance(c, FixAtoms)])
                print(f"固定原子数: {n_fixed}")

        # 初期エネルギー計算
        if self.verbose:
            print("\n初期エネルギー計算中...")
        initial_energy = atoms_opt.get_potential_energy()
        initial_forces = atoms_opt.get_forces()
        initial_fmax = np.max(np.linalg.norm(initial_forces, axis=1))

        if self.verbose:
            print(f"  初期エネルギー: {initial_energy:.4f} eV")
            print(f"  初期最大力: {initial_fmax:.4f} eV/Å")

        # オプティマイザーの初期化
        optimizer_class = self.SUPPORTED_OPTIMIZERS[optimizer]
        opt = optimizer_class(
            atoms_opt,
            trajectory=trajectory_path,
            logfile=logfile,
            **optimizer_kwargs
        )

        # 最適化実行
        if self.verbose:
            print(f"\n{optimizer}最適化実行中...\n")

        opt.run(fmax=fmax, steps=steps)

        # 最終エネルギー
        final_energy = atoms_opt.get_potential_energy()
        final_forces = atoms_opt.get_forces()
        final_fmax = np.max(np.linalg.norm(final_forces, axis=1))

        # 最適化情報
        optimization_info = {
            'optimizer': optimizer,
            'converged': opt.converged(),
            'n_steps': opt.get_number_of_steps(),
            'initial_energy': initial_energy,
            'final_energy': final_energy,
            'energy_change': final_energy - initial_energy,
            'initial_fmax': initial_fmax,
            'final_fmax': final_fmax,
            'fmax_threshold': fmax,
        }

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  最適化完了")
            print(f"{'='*60}")
            print(f"収束: {'✓' if optimization_info['converged'] else '✗'}")
            print(f"ステップ数: {optimization_info['n_steps']}")
            print(f"最終エネルギー: {final_energy:.4f} eV")
            print(f"エネルギー変化: {optimization_info['energy_change']:.4f} eV")
            print(f"最終最大力: {final_fmax:.4f} eV/Å")
            print(f"{'='*60}\n")

        return atoms_opt, optimization_info

    def _fix_bottom_atoms(
        self,
        atoms: Atoms,
        z_threshold: float
    ) -> Atoms:
        """
        Z座標が閾値以下の原子を固定

        Args:
            atoms: Atomsオブジェクト
            z_threshold: Z座標の閾値（最小Z + z_threshold以下を固定）

        Returns:
            制約を設定したAtomsオブジェクト
        """
        z_positions = atoms.positions[:, 2]
        min_z = z_positions.min()
        fixed_indices = [i for i, z in enumerate(z_positions) if z < min_z + z_threshold]

        if fixed_indices:
            constraint = FixAtoms(indices=fixed_indices)
            atoms.set_constraint(constraint)

            if self.verbose:
                print(f"下層原子を固定: {len(fixed_indices)} 個 (z < {min_z + z_threshold:.2f} Å)")

        return atoms


# ========================================================================
# 最適化結果の解析と可視化
# ========================================================================

def analyze_optimization_trajectory(
    trajectory_path: str,
    output_dir: Optional[str] = None,
    plot_filename: Optional[str] = None,
) -> Dict:
    """
    最適化trajectoryを解析してグラフを作成

    Args:
        trajectory_path: Trajectoryファイルのパス
        output_dir: グラフの保存先ディレクトリ（オプション）
        plot_filename: グラフファイル名（オプション）

    Returns:
        解析結果の辞書
    """
    traj = Trajectory(trajectory_path)

    # エネルギーと力の履歴
    energies = []
    fmax_values = []

    for atoms in traj:
        if hasattr(atoms, 'get_potential_energy'):
            try:
                energy = atoms.get_potential_energy()
                energies.append(energy)
            except:
                energies.append(None)

        if hasattr(atoms, 'get_forces'):
            try:
                forces = atoms.get_forces()
                fmax = np.max(np.linalg.norm(forces, axis=1))
                fmax_values.append(fmax)
            except:
                fmax_values.append(None)

    # 有効なデータのみ
    valid_energies = [e for e in energies if e is not None]
    valid_fmax = [f for f in fmax_values if f is not None]

    # 解析結果
    analysis = {
        'n_steps': len(traj),
        'initial_energy': valid_energies[0] if valid_energies else None,
        'final_energy': valid_energies[-1] if valid_energies else None,
        'energy_change': valid_energies[-1] - valid_energies[0] if valid_energies else None,
        'initial_fmax': valid_fmax[0] if valid_fmax else None,
        'final_fmax': valid_fmax[-1] if valid_fmax else None,
    }

    # プロット作成
    if valid_energies or valid_fmax:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # エネルギープロット
        if valid_energies:
            steps = range(len(valid_energies))
            axes[0].plot(steps, valid_energies, 'b-o', markersize=4, linewidth=1.5)
            axes[0].set_xlabel('Optimization Step', fontsize=12)
            axes[0].set_ylabel('Potential Energy (eV)', fontsize=12)
            axes[0].set_title('Energy Convergence', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)

            # エネルギー変化を表示
            if analysis['energy_change'] is not None:
                axes[0].text(
                    0.05, 0.95,
                    f"ΔE = {analysis['energy_change']:.4f} eV",
                    transform=axes[0].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )

        # 力プロット
        if valid_fmax:
            steps = range(len(valid_fmax))
            axes[1].plot(steps, valid_fmax, 'r-o', markersize=4, linewidth=1.5)
            axes[1].set_xlabel('Optimization Step', fontsize=12)
            axes[1].set_ylabel('Max Force (eV/Å)', fontsize=12)
            axes[1].set_title('Force Convergence', fontsize=14, fontweight='bold')
            axes[1].set_yscale('log')
            axes[1].grid(True, alpha=0.3)

            # 収束基準線（0.05 eV/Åの例）
            axes[1].axhline(y=0.05, color='k', linestyle='--', linewidth=1, label='fmax=0.05')
            axes[1].legend()

        plt.tight_layout()

        # 保存
        if output_dir or plot_filename:
            if plot_filename is None:
                plot_filename = Path(trajectory_path).stem + "_analysis.png"
            if output_dir:
                plot_path = Path(output_dir) / plot_filename
            else:
                plot_path = Path(trajectory_path).parent / plot_filename

            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"✓ 解析グラフを保存: {plot_path}")

        plt.close()

    return analysis


# ========================================================================
# 便利な関数
# ========================================================================

def optimize_structure_with_pfp(
    atoms: Atoms,
    output_dir: str = "optimization_results",
    name: str = "structure",
    optimizer: str = 'FIRE',
    fmax: float = 0.05,
    steps: int = 200,
    model_version: str = 'v7.0.0',
    calc_mode: str = 'CRYSTAL_U0',
    fix_bottom_layers: Optional[float] = None,
    save_trajectory: bool = True,
    analyze: bool = True,
) -> Tuple[Atoms, Dict]:
    """
    Matlantis PFPを使った構造最適化の統合関数

    この関数は、最適化の実行、結果の保存、解析を一括で行います。

    Args:
        atoms: 最適化するAtomsオブジェクト
        output_dir: 出力ディレクトリ
        name: 構造の名前（ファイル名に使用）
        optimizer: オプティマイザーの種類
        fmax: 収束判定基準（eV/Å）
        steps: 最大ステップ数
        model_version: Matlantisモデルバージョン
        calc_mode: 計算モード
        fix_bottom_layers: 下層固定の閾値（オプション）
        save_trajectory: Trajectoryを保存するか
        analyze: 最適化後に解析を実行するか

    Returns:
        (optimized_atoms, results): 最適化後の構造と結果の辞書
    """
    # 出力ディレクトリの作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ファイルパス
    traj_path = str(output_path / f"{name}_optimization.traj") if save_trajectory else None
    xyz_path = output_path / f"{name}_optimized.xyz"

    # オプティマイザーの初期化
    opt_engine = MatlantisOptimizer(
        model_version=model_version,
        calc_mode=calc_mode,
        verbose=True
    )

    # 最適化実行
    optimized_atoms, opt_info = opt_engine.optimize(
        atoms=atoms,
        optimizer=optimizer,
        fmax=fmax,
        steps=steps,
        trajectory_path=traj_path,
        fix_bottom_layers=fix_bottom_layers,
    )

    # 最適化後の構造を保存
    write(str(xyz_path), optimized_atoms)
    print(f"✓ 最適化構造を保存: {xyz_path}")

    # 解析
    analysis_results = {}
    if analyze and traj_path:
        analysis_results = analyze_optimization_trajectory(
            trajectory_path=traj_path,
            output_dir=str(output_path),
        )

    # 統合結果
    results = {
        'optimization_info': opt_info,
        'analysis': analysis_results,
        'output_files': {
            'trajectory': traj_path,
            'optimized_structure': str(xyz_path),
        }
    }

    return optimized_atoms, results


# ========================================================================
# 使用例（テスト用）
# ========================================================================

def example_usage():
    """使用例"""
    from ase.build import bulk

    print("\n" + "="*70)
    print("  Matlantis PFP 最適化ユーティリティの使用例")
    print("="*70 + "\n")

    # テスト構造（Al結晶）
    atoms = bulk('Al', 'fcc', a=4.05).repeat((2, 2, 2))
    atoms.rattle(stdev=0.1)  # 少しランダムに揺らす

    print(f"テスト構造: {atoms.get_chemical_formula()}")
    print(f"原子数: {len(atoms)}\n")

    # 最適化実行
    optimized_atoms, results = optimize_structure_with_pfp(
        atoms=atoms,
        output_dir="test_optimization",
        name="al_bulk",
        optimizer='FIRE',
        fmax=0.05,
        steps=100,
    )

    print("\n最適化完了！")
    print(f"収束: {results['optimization_info']['converged']}")
    print(f"エネルギー変化: {results['optimization_info']['energy_change']:.4f} eV")


if __name__ == "__main__":
    example_usage()
