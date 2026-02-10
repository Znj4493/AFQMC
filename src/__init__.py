"""
AFQMC 蚂蚁金服语义相似度匹配项目

src 模块初始化文件
"""

__version__ = "0.1.0"
__author__ = "AFQMC Team"

# 导出常用模块，方便使用
from .data.data_loader import (
    load_jsonl,
    load_train_data,
    load_test_data,
    get_data_statistics,
    split_train_val,
    display_samples
)

__all__ = [
    'load_jsonl',
    'load_train_data',
    'load_test_data',
    'get_data_statistics',
    'split_train_val',
    'display_samples',
]
