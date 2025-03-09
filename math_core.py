import math
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


# ======================
# 3. 测试数据生成
# ======================
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


def analyze_results(model, data):
    """执行可视化分析"""
    # 准备数据
    diffs = []
    preds = []
    for s, p, ad in data:
        diffs.append(ad)
        preds.append(model.calculate_diff(s, p, ad))
    
    # 创建画布
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 散点分布
    axs[0,0].scatter(diffs, preds, alpha=0.6)
    axs[0,0].plot([0,1], [0,1], 'r--')
    axs[0,0].set_title('实际难度 vs 预测难度')
    axs[0,0].set_xlabel('实际绝对难度')
    axs[0,0].set_ylabel('预测相对难度')
    
    # 2. 残差分析
    residuals = np.array(preds) - np.array(diffs)
    axs[0,1].hist(residuals, bins=20, edgecolor='black')
    axs[0,1].set_title('预测残差分布')
    
    # 3. 维度分析
    dims = ['语法', '逻辑', '细节', '词汇']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, dim in enumerate(dims):
        x = [p[dim]-s[dim] for s,p,_ in data]
        y = residuals
        axs[1,0].scatter(x, y, c=colors[i], alpha=0.5, label=dim)
    axs[1,0].axhline(0, c='gray', ls='--')
    axs[1,0].set_title('各维度差异对残差的影响')
    axs[1,0].legend()
    
    # 4. 3D可视化
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.scatter(
        [s['语法'] for s,_,_ in data],
        [s['逻辑'] for s,_,_ in data],
        residuals,
        c=residuals,
        cmap='viridis'
    )
    ax.set_title('语法-逻辑能力残差分布')
    ax.set_xlabel('语法能力')
    ax.set_ylabel('逻辑能力')
    ax.set_zlabel('残差')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    model = EnhancedDifficultyModel()
    test_data = generate_test_data(500)
    model.train(test_data, epochs=100)
    analyze_results(model, test_data)
    sample_case = test_data[0]
    print("\n示例预测:")
    print(f"绝对难度: {sample_case[2]:.2f}")
    print(f"预测难度: {model.calculate_diff(*sample_case):.2f}")