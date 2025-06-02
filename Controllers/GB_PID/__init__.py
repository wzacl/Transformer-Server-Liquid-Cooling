"""
GB_PID 控制器模塊

此模塊實現了保證有界PID控制器，用於控制冷卻系統中的風扇和泵。

模塊包含：
- GB_PID_fan: 風扇控制器
- GB_PID_pump: 泵控制器
"""

# 從各自的模塊中導入類
from .GB_PID_fan import GB_PID_fan
from .GB_PID_pump import GB_PID_pump

__all__ = ['GB_PID_fan', 'GB_PID_pump'] 