import sys
sys.path.append('src')
from utils import compute_metrics
import numpy as np

# 模拟你的实际数据
y_true = np.array([0]*2211 + [1]*989)
# 根据混淆矩阵构造预测结果
y_pred = np.array([0]*1746 + [1]*465 + [0]*380 + [1]*609)

# 打印详细报告
compute_metrics(y_true, y_pred, verbose=True)
