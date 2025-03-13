import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedDifficultyModel:
    def __init__(self):
        self.cog_net = nn.Sequential(
            nn.Linear(8, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.optimizer = optim.AdamW(self.cog_net.parameters(), lr=0.005)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.criterion = self.quantile_loss
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.best_loss = float('inf')

    def quantile_loss(self, pred, target):
        residual = target - pred
        return torch.mean(torch.where(residual > 0, 0.3*residual, 0.7*(-residual)))

    def train(self, data, epochs=100):
        X = np.array([list(s.values())+list(p.values()) for s,p,_ in data])
        y = np.array([ad for _,_,ad in data]).reshape(-1,1)
        
        # 标准化
        self.scaler.fit(X)
        self.y_scaler.fit(y)
        X_scaled = self.scaler.transform(X)
        y_scaled = self.y_scaler.transform(y)
        
        # 训练循环（确保批次大小≥2）
        batch_size = 16
        num_samples = len(X_scaled)
        indices = np.random.permutation(num_samples)
        
        for epoch in range(epochs):
            self.cog_net.train()
            total_loss = 0        
            # 按批次训练（跳过最后不完整批次）
            for i in range(0, num_samples - batch_size + 1, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = torch.FloatTensor(X_scaled[batch_indices])
                batch_y = torch.FloatTensor(y_scaled[batch_indices])
                
                self.optimizer.zero_grad()
                outputs = self.cog_net(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.cog_net.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
            
            self.scheduler.step()
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/(num_samples//batch_size):.4f}")

    def calculate_diff(self, student, problem, abs_diff):
        input_data = np.array([list(student.values()) + list(problem.values())])
        input_scaled = self.scaler.transform(input_data)
        with torch.no_grad():
            pred_scaled = self.cog_net(torch.FloatTensor(input_scaled)).item()
        return round(self.y_scaler.inverse_transform([[pred_scaled]])[0][0], 2)
    

def generate_test_data(n=1000):
    """生成模拟数据"""
    np.random.seed(42)
    data = []
    for _ in range(n):
        student = {
            '语法': np.clip(np.random.beta(2,3), 0.05, 0.95),
            '逻辑': np.clip(np.random.beta(2,3), 0.05, 0.95),
            '细节': np.clip(np.random.beta(2,3), 0.05, 0.95),
            '词汇': np.clip(np.random.beta(2,3), 0.05, 0.95)
        }
        
        problem = {
            k: np.clip(v + np.random.choice([-0.2,0.2]), 0.05, 0.95)
            for k, v in student.items()
        }
        
        abs_diff = sum(problem.values()) / 4
        data.append((student, problem, abs_diff))
    return data
    

class DifficultyPredictor:
    """封装好的难度预测器"""
    def __init__(self):
        self.cog_net = self._build_model()
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def _build_model(self) -> nn.Module:
        """构建模型结构"""
        return nn.Sequential(
            nn.Linear(8, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def save(self, filepath: str):
        if not hasattr(self.scaler, 'mean_'):
            raise RuntimeError("必须先用训练数据调用fit()方法")
        torch.save({
            'model_state': self.cog_net.state_dict(),
            # 保存scaler完整参数
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_,  # 新增关键参数
            'scaler_var': self.scaler.var_,
            'y_scaler_mean': self.y_scaler.mean_,
            'y_scaler_scale': self.y_scaler.scale_,  # 新增关键参数
            'y_scaler_var': self.y_scaler.var_
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'DifficultyPredictor':
        instance = cls()
        try:
            checkpoint = torch.load(filepath, weights_only=False)
        except:
            checkpoint = torch.load(filepath, weights_only=True)
        # 恢复模型参数
        instance.cog_net.load_state_dict(checkpoint['model_state'])
        
        # 重构scaler
        instance.scaler.mean_ = checkpoint['scaler_mean']
        instance.scaler.var_ = checkpoint['scaler_var']
        instance.scaler.scale_ = checkpoint['scaler_scale']  # 使用正确字段
        
        # 重构y_scaler
        instance.y_scaler.mean_ = checkpoint['y_scaler_mean']
        instance.y_scaler.var_ = checkpoint['y_scaler_var']
        instance.y_scaler.scale_ = checkpoint['y_scaler_scale']
        
        return instance

    def predict(self, student: Dict[str, float], problem: Dict[str, float]) -> float:
        """
        对外暴露的预测接口
        :param student: 学生能力字典，包含语法/逻辑/细节/词汇四个维度
        :param problem: 题目特征字典，包含语法/逻辑/细节/词汇四个维度
        :return: 预测的相对难度值
        """
        # 转换为模型输入格式
        input_data = np.array([
            list(student.values()) + list(problem.values())
        ])
        # 数据标准化
        input_scaled = self.scaler.transform(input_data)
        with torch.no_grad():
            pred_scaled = self.cog_net(torch.FloatTensor(input_scaled)).item()
        # 反标准化
        return round(self.y_scaler.inverse_transform([[pred_scaled]])[0][0], 2)

# 使用示例 ---------------------------------------------------------------------
if __name__ == "__main__":
    # 训练原始模型
    original_model = EnhancedDifficultyModel()
    test_data = generate_test_data(500)
    original_model.train(test_data, epochs=100)
    
    # 创建封装器并继承参数
    predictor = DifficultyPredictor()
    predictor.cog_net.load_state_dict(original_model.cog_net.state_dict())  # 加载模型参数
    predictor.scaler = original_model.scaler  # 继承已拟合的scaler
    predictor.y_scaler = original_model.y_scaler
    
    predictor.save("difficulty_model.pth")
