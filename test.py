import numpy as np
import main, fight
import heapq

a = np.array(
    [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, -1, 0, 0, 0], [0, 0, 0, -1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])

import time

begin = time.time()
my_color = 1
ai1 = main.AI(chessboard_size=8, color=my_color, time_out=5)
ai2 = fight.AI(chessboard_size=8, color=-my_color, time_out=5)
print(a)
while True:
    begin1 = time.time()
    list1 = ai1.go(a)
    if len(list1) != 0:
        x, y = list1[-1]
        print(list1)
        a = main.update_chessboard(x, y, a, ai1.color)
    end = time.time()
    list2 = ai2.go(a)
    if len(list2) != 0:
        x, y = list2[-1]
        a = main.update_chessboard(x, y, a, ai2.color)
    if len(list1) == 0 and len(list2) == 0:
        print(len(np.where(a == my_color)[0]))
        if len(np.where(a == my_color)[0]) > 32:
            print('You lose!')
        if len(np.where(a == my_color)[0]) < 32:
            print('You win!')
        break
    print(a)
    print('time: ' + str(end - begin1))
    print('--------------------------------------')
