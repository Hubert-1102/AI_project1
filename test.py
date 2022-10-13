import numpy as np
import main
import heapq
a = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, -1, -1, -1, 1, 0, 0],
    [0, 1, 1, -1, -1, 0, 0, 0],
    [0, 0, -1, -1, -1, -1, 0, 0],
    [0, -1, 1, 1, 1, 1, 0, 0],
    [-1, 0, -1, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]])
import time
begin = time.time()
ai = main.AI(chessboard_size=8, color=-1, time_out=5)
print(a)
list0=ai.go(a)
print(list0[-1])
print(time.time()-begin)
list1=ai.go_greedy(a)
print(list1[-1])
# print(a[3])
