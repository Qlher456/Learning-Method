# Learning-Method

# Combined.py

使用自监督学习预训练20次，再使用主动学习迭代50次

# 超参数

batch_size = 64

learning_rate = 0.001

pretrain_epochs = 20

num_epochs = 100

initial_label_percent = 0.1

query_percent_per_cycle = 0.05

num_cycles = 5


# 实验展示

Pretrain Epoch [1/20] Loss: 0.8421

Pretrain Epoch [5/20] Loss: 0.7365

Pretrain Epoch [10/20] Loss: 0.7303

Pretrain Epoch [15/20] Loss: 0.7207

Pretrain Epoch [20/20] Loss: 0.7138

Active Learning Cycle 1/5

Cycle 1, Epoch [1/20] Train Loss: 2.0612, Train Acc: 25.20% Test Loss: 1.9531, Test Acc: 29.58%

Cycle 1, Epoch [20/20] Train Loss: 1.8282, Train Acc: 34.12% Test Loss: 1.8663, Test Acc: 32.12%

Active Learning Cycle 2/5

Cycle 2, Epoch [1/20] Train Loss: 1.9478, Train Acc: 28.57% Test Loss: 1.8692, Test Acc: 31.87%

Cycle 2, Epoch [20/20] Train Loss: 1.9035, Train Acc: 31.02% Test Loss: 1.8570, Test Acc: 33.13%

Active Learning Cycle 3/5

Cycle 3, Epoch [1/20] Train Loss: 1.9775, Train Acc: 27.64% Test Loss: 1.8554, Test Acc: 33.29%

Cycle 3, Epoch [20/20] Train Loss: 1.9501, Train Acc: 28.17% Test Loss: 1.8656, Test Acc: 33.07%

Active Learning Cycle 4/5

Cycle 4, Epoch [1/20] Train Loss: 1.9876, Train Acc: 27.22% Test Loss: 1.8603, Test Acc: 32.50%

Cycle 4, Epoch [20/20] Train Loss: 1.9680, Train Acc: 28.31% Test Loss: 1.8534, Test Acc: 33.59%

Active Learning Cycle 5/5

Cycle 5, Epoch [1/20] Train Loss: 1.9926, Train Acc: 27.27% Test Loss: 1.8452, Test Acc: 33.80%

Cycle 5, Epoch [20/20] Train Loss: 1.9735, Train Acc: 28.33% Test Loss: 1.8512, Test Acc: 33.74%

#  Active-Learning

Epoch [1/100] Train Loss: 2.2354, Train Acc: 15.00% Test Loss: 2.0990, Test Acc: 26.32%
