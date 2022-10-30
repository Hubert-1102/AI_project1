import numpy as np
import main, fight, greedy_fight
import heapq
import time


def play(black, white, chessboard):
    a = chessboard
    ai1 = black
    ai2 = white
    print(a)
    while True:
        begin1 = time.time()
        list1 = ai1.go(a)
        if len(list1) != 0:
            print(str(len(list1))+'len')
            x, y = list1[-1]
            # print(list1)
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
                return 0
            if len(np.where(a == my_color)[0]) < 32:
                print('You win!')
                return 1

            break
        print(a)
        print(len(np.where(a == my_color)[0]) - len(np.where(a == -my_color)[0]))
        print('time: ' + str(end - begin1))
        print('--------------------------------------')


a = np.array(
    [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, -1, 0, 0, 0], [0, 0, 0, -1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])

begin = time.time()
my_color = 1
count_win = 0
total = 2
for i in range(total):
    print('new game')
    result = 0
    if i % 2 == 0:
        my_color = 1
        fight_ai = fight.AI(chessboard_size=8, color=-my_color, time_out=5)
        me = main.AI(chessboard_size=8, color=my_color, time_out=5)
        result = play(black=fight_ai, white=me, chessboard=a)
    else:
        my_color = -1
        fight_ai = fight.AI(chessboard_size=8, color=-my_color, time_out=5)
        me = main.AI(chessboard_size=8, color=my_color, time_out=5)
        result = play(black=me, white=fight_ai, chessboard=a)
    count_win += result
    fight.round = 0
    main.round = 0
    a = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, -1, 0, 0, 0], [0, 0, 0, -1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
print('current version wins: ' + str(count_win))
print('total: ' + str(total))
