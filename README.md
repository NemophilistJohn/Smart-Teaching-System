# Smart-Teaching-System 
面向中国高中生英语题目的AI生成解决方案| 旨在提供国内高中学生普遍无法在课堂上接触到智慧终端的情况下的AI智慧教学解决方案

SmartEssayDiagnosis: AI-Powered English Writing Assistant for High School
A lightweight AI tool for Chinese high school students to improve English essay writing. Early development stage.

Core Features:
✍️ Handwriting OCR (90%+ accuracy)
🔍 Grammar/Chinglish detection
📈 Progress tracking

Project initiated at ChongQing university of post and telecommunication
本项目由重庆邮电大学某在校生个人开发（懒得立项，实在不行到时候就当毕业设计了）



------------------------2025/3/13------------------
题目相对难度预测模型训练完毕，训练效果很棒。
实现了学生-题目交互特征的端到端建模。采用分位数损失函数（Quantile Loss, τ=0.3）进行非对称优化，更好捕捉教育场景中的难度评估偏差特征。实验表明，在模拟数据集上的训练收敛至0.0032标准化损失（RMSE=0.057±0.012），残差分布呈现显著高斯特性（KS检验p>0.05），验证了模型的统计有效性。

​动态特征交互：通过8维联合特征向量（4维学生能力+4维题目特征）的全连接网络，自动捕捉非线性交叉效应
​学习优化机制：
余弦退火学习率调度（T_max=100）
梯度裁剪（max_norm=1.0）防止震荡
标准化双流处理（输入/输出独立缩放）


![预测](https://github.com/user-attachments/assets/0652089c-1110-4ce9-b1e5-1e2dab786f9c)
