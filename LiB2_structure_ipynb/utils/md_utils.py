"""
MD (分子動力学) シミュレーション関連のユーティリティ関数

このモジュールには以下の機能が含まれます：
- Integrator選択とMD実行
- MDログの記録
"""

import logging
from time import perf_counter
from typing import Dict, Any

import numpy as np
from ase import units

from matlantis_features.features.md import (
    MDFeature,
    ASEMDSystem,
    MDExtensionBase,
    VelocityVerletIntegrator,
    LangevinIntegrator,
    NVTBerendsenIntegrator,
    NPTBerendsenIntegrator,
    NPTIntegrator,
)
from matlantis_features.utils.calculators import pfp_estimator_fn


class PrintWriteLog(MDExtensionBase):
    """
    MDシミュレーションのログを記録・出力するクラス

    Args:
        fname (str): ログファイル名（拡張子なし）
        dirout (str): 出力ディレクトリ（デフォルト: '.'）
        stdout (bool): 標準出力に表示するか（デフォルト: False）
    """

    def __init__(self, fname: str, dirout: str = '.', stdout: bool = False):
        self.fname = fname
        self.dirout = dirout
        self.t_start = perf_counter()
        self.stdout = stdout

    def __call__(self, system, integrator):
        n_step = system.current_total_step
        sim_time = system.current_total_time / 1000  # ps
        E_tot = system.ase_atoms.get_total_energy()
        E_pot = system.ase_atoms.get_potential_energy()
        E_kin = system.ase_atoms.get_kinetic_energy()
        temp = system.ase_atoms.get_temperature()
        density = system.ase_atoms.get_masses().sum() / units.mol / (
            system.ase_atoms.cell.volume * (1e-8**3)
        )
        calc_time = (perf_counter() - self.t_start) / 60.  # min.

        # ヘッダー書き込み
        if n_step == 0:
            hdr = 'step,time[ps],E_tot[eV],E_pot[eV],E_kin[eV],'
            hdr += 'T[K],density[g/cm3],calc_time[min]'
            with open(f'{self.dirout}/{self.fname}.log', 'w') as f_log:
                f_log.write(f'{hdr}\n')

        # 結果の書き込みと出力
        line = f'{n_step:8d},{sim_time:7.2f},'
        line += f'{E_tot:11.4f},{E_pot:11.4f},{E_kin:9.4f},'
        line += f'{temp:8.2f},{density:7.3f},{calc_time:8.2f}'
        with open(f'{self.dirout}/{self.fname}.log', 'a') as f_log:
            f_log.write(f'{line}\n')
        if self.stdout:
            print(line)


def select_integrator(
    integrator_type: str,
    timestep: float,
    temperature: float,
    pressure: float = 101325.0,
):
    """
    指定された文字列に応じて、対応するIntegratorオブジェクトを返す

    Args:
        integrator_type (str): Integratorの種類
            - "NVE": Velocity Verlet (エネルギー保存)
            - "Langevin": Langevin dynamics (NVT)
            - "NVT_Berendsen": Berendsen thermostat (NVT)
            - "NPT_Berendsen": Berendsen thermostat and barostat (NPT)
            - "NPT": Nosé-Hoover thermostat and Parrinello-Rahman barostat (NPT)
        timestep (float): タイムステップ (fs)
        temperature (float): 温度 (K)
        pressure (float): 圧力 (Pa)、NPT系でのみ使用

    Returns:
        Integratorオブジェクト
    """
    print(f"--- Integrator: '{integrator_type}' を選択しました ---")

    # 圧力をASE unitsに変換
    pressure_au = pressure * units.Pascal

    if integrator_type == "NVE":
        return VelocityVerletIntegrator(timestep=timestep)

    elif integrator_type == "Langevin":
        return LangevinIntegrator(
            timestep=timestep,
            temperature=temperature,
            friction=0.002
        )

    elif integrator_type == "NVT_Berendsen":
        return NVTBerendsenIntegrator(
            timestep=timestep,
            temperature=temperature,
            taut=100.0 * units.fs
        )

    elif integrator_type == "NPT_Berendsen":
        return NPTBerendsenIntegrator(
            timestep=timestep,
            temperature=temperature,
            pressure=pressure_au,
            taut=100.0 * units.fs,
            taup=1000.0 * units.fs
        )

    elif integrator_type == "NPT":
        return NPTIntegrator(
            timestep=timestep,
            temperature=temperature,
            pressure=pressure_au,
            ttime=25.0 * units.fs,
            pfactor=75.0 * units.fs
        )

    else:
        raise ValueError(f"エラー: 指定されたIntegrator '{integrator_type}' は不正です。")


def run_md_simulation(
    atoms,
    integrator_type: str,
    temperature: float,
    n_steps: int,
    timestep: float = 1.0,
    pressure: float = 101325.0,
    traj_file: str = "md.traj",
    traj_freq: int = 50,
    logger_interval: int = 50,
    model_version: str = 'v7.0.0',
    calc_mode: str = 'CRYSTAL_U0',
    show_progress: bool = True,
    show_logger: bool = True,
):
    """
    MDシミュレーションを実行する汎用関数

    Args:
        atoms: ASE Atomsオブジェクト
        integrator_type (str): Integratorの種類
        temperature (float): 温度 (K)
        n_steps (int): 実行ステップ数
        timestep (float): タイムステップ (fs)
        pressure (float): 圧力 (Pa)
        traj_file (str): トラジェクトリファイル名
        traj_freq (int): トラジェクトリ保存間隔
        logger_interval (int): ログ出力間隔
        model_version (str): PFPモデルバージョン
        calc_mode (str): PFP計算モード
        show_progress (bool): 進捗バー表示
        show_logger (bool): ログ表示

    Returns:
        MDシミュレーション結果
    """
    # ロガー設定
    logger = logging.getLogger("matlantis_features")
    logger.setLevel(logging.INFO)

    # PFP計算器の準備
    estimator_fn = pfp_estimator_fn(model_version=model_version, calc_mode=calc_mode)

    # MDシステムの初期化
    system = ASEMDSystem(atoms)
    system.init_temperature(
        temperature=temperature,
        stationary=True,
        rng=np.random.RandomState(seed=12345)
    )
    print(f"システムを {temperature} K に初期化しました。")

    # Integratorの選択と準備
    integrator = select_integrator(integrator_type, timestep, temperature, pressure)

    # MDFeatureの設定
    md = MDFeature(
        integrator=integrator,
        n_run=n_steps,
        show_progress_bar=show_progress,
        show_logger=show_logger,
        logger_interval=logger_interval,
        traj_file_name=traj_file,
        traj_freq=traj_freq,
        estimator_fn=estimator_fn,
    )

    # MDシミュレーションの実行
    print(f"\n--- MDシミュレーションを開始します ({n_steps} ステップ) ---")
    md_results = md(system)

    print("\n--- MDシミュレーションが正常に完了しました ---")
    print(f"最終的なトラジェクトリは '{traj_file}' に保存されました。")
    print(f"最終ステップ: {system.current_total_step}, 最終時間: {system.current_time:.2f} fs")

    return md_results


def run_constant_temp_md(md_params: Dict[str, Any]):
    """
    一定温度でNVT-MDを実行するための関数

    Args:
        md_params (dict): MDパラメータを含む辞書
            - atoms: ASE Atomsオブジェクト
            - temperature: 温度 (K)
            - timestep: タイムステップ (fs)
            - n_run: 実行ステップ数
            - model_version: PFPモデルバージョン
            - calc_mode: PFP計算モード
            - dirout: 出力ディレクトリ
            - fname: ファイル名
            - logger_interval: ログ間隔
            - traj_freq: トラジェクトリ保存間隔

    Returns:
        MDシミュレーション結果
    """
    estimator_fn = pfp_estimator_fn(
        model_version=md_params['model_version'],
        calc_mode=md_params['calc_mode'],
    )
    system = ASEMDSystem(
        atoms=md_params['atoms'],
        step=0,
        time=0.0,
    )
    integrator = NVTBerendsenIntegrator(
        timestep=md_params['timestep'],
        temperature=md_params['temperature'],
        taut=100.,
        fixcm=True,
    )
    md = MDFeature(
        integrator=integrator,
        n_run=md_params['n_run'],
        show_progress_bar=True,
        show_logger=False,
        logger_interval=md_params['logger_interval'],
        estimator_fn=estimator_fn,
        traj_file_name=f"{md_params['dirout']}/{md_params['fname']}.traj",
        traj_freq=md_params['traj_freq'],
    )

    print(f"\n--- MD計算を開始: {md_params['fname']} ---")
    print(f"ステップ数: {md_params['n_run']} ({md_params['n_run'] * md_params['timestep']/1000:.1f} ps), 温度: {md_params['temperature']:.2f} K")

    md_results = md(
        system=system,
        extensions=[
            (PrintWriteLog(fname=md_params['fname'], dirout=md_params['dirout'], stdout=True),
             md_params['logger_interval'])
        ]
    )
    print(f"--- MD計算が完了: {md_params['fname']} ---")
    return md_results
