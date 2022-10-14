import numpy as np
import main
import heapq

a = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, -1, 1, 0, 0, 0], [0, 0, 0, -1, 1, 0, 0, 0],
    [0, 0, 0, -1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]])

# a= np.ones((8,8))
# a[7,0] = 0
# a[7,7]=-1
import time

begin = time.time()
ai = main.AI(chessboard_size=8, color=-1, time_out=5)
print(a)
list0 = ai.go(a)
print(list0)

print(time.time() - begin)
list1 = ai.go1(a)
print(list1)
# print(a[3])
