# Learning-Method

# Combined.py

使用自监督学习预训练20次，再使用主动学习迭代50次

# 超参数

batch_size = 64

learning_rate = 0.001

num_epochs = 50

pretrain_epochs = 20

initial_label_percent = 0.1  # 初始标注数据比例

query_percent_per_cycle = 0.05  # 每次主动学习的查询比例

num_cycles = 5

# 实验展示

Pretrain Epoch [1/20] Loss: 2.9279

