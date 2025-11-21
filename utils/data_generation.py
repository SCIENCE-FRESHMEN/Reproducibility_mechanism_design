import torch
import numpy as np


def generate_valuations(num_samples, num_buyers, num_items, dist='uniform', seed=42, device='cpu'):
    """生成买家对物品的估值数据"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if dist == 'uniform':
        return torch.rand(num_samples, num_buyers, num_items).to(device)
    elif dist == 'normal':
        vals = torch.randn(num_samples, num_buyers, num_items).to(device)
        return torch.clamp(vals, 0, 1)  # 限制在[0,1]范围内
    else:
        raise ValueError("不支持的分布类型")