"""
選股因子模組
提供選股因子基類和實作
"""

from .base_selection_factor import StockPickerBase
from .regress_angle_factor import RegressAnglePicker
from .rps_factor import RPSPicker

__all__ = [
    'StockPickerBase',
    'RegressAnglePicker',
    'RPSPicker',
]

