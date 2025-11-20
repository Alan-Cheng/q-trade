"""
滑價模組
提供滑價模型基類和實作
"""

from .base_slippage import BaseSlippage
from .open_gap_slippage import SlippageOpenGap

__all__ = [
    'BaseSlippage',
    'SlippageOpenGap',
]

