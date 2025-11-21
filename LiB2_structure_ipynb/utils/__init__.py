"""
LiB2構造シミュレーション用のユーティリティモジュール

このモジュールは、ipynbファイル間で共通するコードをまとめたものです。

サブモジュール:
- md_utils: MD関連の関数とクラス
- optimization_utils: 構造最適化関連
- structure_utils: 構造操作関連
- analysis_utils: 解析関連
- io_utils: ファイルI/O関連
"""

# MD関連
from .md_utils import (
    PrintWriteLog,
    select_integrator,
    run_md_simulation,
    run_constant_temp_md,
)

# 最適化関連
from .optimization_utils import (
    run_matlantis_optimization,
    MatlantisOptimizer,
    optimize_structure_with_pfp,
    analyze_optimization_trajectory,
)

# 構造操作関連
from .structure_utils import (
    build_interface,
    calculate_density,
    calculate_cell_from_density,
    set_cell_with_vacuum,
    create_water_unit_cell,
)

# 解析関連
from .analysis_utils import (
    count_fragms,
    find_unexpected_molecules,
    analyze_molecular_evolution_and_save,
    analyze_trajectory_batch,
)

# ファイルI/O関連
from .io_utils import (
    convert_traj_to_cif,
    clean_small_traj_files,
    batch_convert_traj_to_cif,
)

__all__ = [
    # MD関連
    'PrintWriteLog',
    'select_integrator',
    'run_md_simulation',
    'run_constant_temp_md',
    # 最適化関連
    'run_matlantis_optimization',
    'MatlantisOptimizer',
    'optimize_structure_with_pfp',
    'analyze_optimization_trajectory',
    # 構造操作関連
    'build_interface',
    'calculate_density',
    'calculate_cell_from_density',
    'set_cell_with_vacuum',
    'create_water_unit_cell',
    # 解析関連
    'count_fragms',
    'find_unexpected_molecules',
    'analyze_molecular_evolution_and_save',
    'analyze_trajectory_batch',
    # ファイルI/O関連
    'convert_traj_to_cif',
    'clean_small_traj_files',
    'batch_convert_traj_to_cif',
]

__version__ = '1.0.0'
