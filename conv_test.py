import numpy as np
import scipy 
# 定义输入信号和卷积核




with open('array.txt', 'r') as file:
    lines = file.readlines()
    array = [list(map(int, line.strip().split(','))) for line in lines]


# 卷积核需要翻转
with open('kernel.txt', 'r') as file:
    lines = file.readlines()
    kernel = [list(map(int, line.strip().split(','))) for line in lines]
kernel_90 = np.flip(kernel,axis = 0)
kernel_180 = np.flip(kernel_90,axis = 1)

# 使用scipy.signal.convolve2d函数进行卷积操作
y = scipy.signal.convolve2d(array, kernel_180, mode='valid')

with open('result.txt', 'r') as file:
    lines = file.readlines()
    result = [list(map(int, line.strip().split(','))) for line in lines]


if np.array_equal(y, result):
    print("卷积结果正确")
else:
    print("卷积结果错误")

#print(y)