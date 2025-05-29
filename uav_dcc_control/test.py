import numpy as np
size = 400
batch_size = 100

# 随机选择起始点，确保连续范围不超出 size
start = np.random.randint(0, size - batch_size + 1)
ind = np.arange(start, start + batch_size)
print(ind)