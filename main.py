import numpy as np
import random
import time, heapq
import numba as nb
from queue import PriorityQueue

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(1)
access_number: int = 5000
access_time: int = 5
heap = []
max_access = 0
best_p = (-1, -1)


class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []

    def go(self, chessboard):
        self.candidate_list.clear()
        begin = time.time()
        self.candidate_list += next_moves(chessboard, self.color, False)
        rootNode: Node = Node(parent=None, chessboard=chessboard, color=self.color, x=-1, y=-1)
        root = Node(parent=None, chessboard=chessboard, color=self.color, x=-1, y=-1)
        for i in range(access_number):
            # print(i)
            node: Node = tree_policy(rootNode,root)
            reward = default_policy(node.chessboard, node.color)
            backup(node, reward, rootNode)
            if i % 40 == 0:
                self.candidate_list.append(best_p)
            if time.time()-begin>self.time_out-0.2:
                return self.candidate_list
        # best_node = best_child(rootNode)
        return self.candidate_list

    def go_greedy(self, chessboard):
        self.candidate_list.clear()
        # write algorithm
        idx = np.where(chessboard == COLOR_NONE)
        idx_list = list(zip(idx[0], idx[1]))
        queue = PriorityQueue()
        list1 = []
        all_edge = 0
        for x, y in idx_list:
            piece_number, x_distance, y_distance = valid_position(chessboard, x, y, self.color)
            list1.append((x, y, piece_number, x_distance, y_distance))
            if piece_number > 0 and x_distance < 3.5 and y_distance < 3.5:
                all_edge = 2
        for x, y, piece_number, x_distance, y_distance in list1:
            # piece_number, x_distance, y_distance = self.valid_position(chessboard, x, y, self.color)
            if piece_number > 0:
                self.candidate_list.append((x, y))
                para = piece_number + x_distance + y_distance
                if x_distance == 3.5 and y_distance == 3.5:
                    para = para * 20
                else:
                    if x_distance == 3.5 and y_distance != 2.5:
                        para = para + all_edge
                    if y_distance == 3.5 and x_distance != 2.5:
                        para = para + all_edge
                    if x_distance == 3.5 and y_distance == 2.5:
                        para = para - 2
                    if y_distance == 3.5 and x_distance == 2.5:
                        para = para - 2
                    if x_distance == 2.5 and y_distance == 2.5:
                        para = para / 1.5
                queue.put((para, (x, y)))
        if queue.empty():
            return []
        para, position = queue.get()
        self.candidate_list.append(position)
        return self.candidate_list


@nb.jit(nopython=True)
def valid_position(chessboard, x, y, color):
    # 竖直方向
    count = 0

    t = 0
    while x + t + 1 <= 7 and chessboard[x + t + 1][y] == -1 * color:
        t += 1
    if x + t + 1 <= 7 and chessboard[x + t + 1][y] == color:
        count += t

    t = 0
    while x - t - 1 >= 0 and chessboard[x - 1 - t][y] == -1 * color:
        t += 1
    if x - t - 1 >= 0 and chessboard[x - t - 1][y] == color:
        count += t

    # 水平方向

    t = 0
    while y + t + 1 <= 7 and chessboard[x][y + t + 1] == -1 * color:
        t += 1
    if y + t + 1 <= 7 and chessboard[x][y + t + 1] == color:
        count += t

    t = 0
    while y - t - 1 >= 0 and chessboard[x][y - t - 1] == -1 * color:
        t += 1
    if y - t - 1 >= 0 and chessboard[x][y - t - 1] == color:
        count += t
    # 对角方向     \

    t = 0
    while x + t + 1 <= 7 and y + 1 + t < 7 and chessboard[x + 1 + t][
        y + 1 + t] == -1 * color:
        t += 1
    if x + t + 1 <= 7 and y + 1 + t <= 7 and chessboard[x + t + 1][y + t + 1] == color:
        count += t

    t = 0
    while x - t - 1 >= 0 and y - t - 1 > 0 and chessboard[x - t - 1][y - t - 1] == -1 * color:
        t += 1
    if x - t - 1 >= 0 and y - t - 1 >= 0 and chessboard[x - t - 1][y - t - 1] == color:
        count += t

    # 对角方向     /
    t = 0
    while x + 1 + t <= 7 and y - t - 1 > 0 and chessboard[x + t + 1][
        y - t - 1] == -1 * color:
        t += 1
    if x + 1 + t <= 7 and y - t - 1 >= 0 and chessboard[x + 1 + t][y - t - 1] == color:
        count += t

    t = 0
    while x - t - 1 >= 0 and y + t + 1 < 7 and chessboard[x - t - 1][
        y + t + 1] == -1 * color:
        t += 1
    if x - t - 1 >= 0 and y + t + 1 <= 7 and chessboard[x - t - 1][y + t + 1] == color:
        count += t
    return count, abs(x - 3.5), abs(y - 3.5)


class Node(object):
    def __init__(self, parent=None, chessboard=None, color=None, x=-1, y=-1):
        self.parent = parent
        self.chessboard: np.ndarray = chessboard
        self.children = {}
        self.color: int = color
        self.access = 0
        self.reward = 0
        self.x = x
        self.y = y

    def expand(self, x, y):
        chessboard2 = self.chessboard.copy()
        update_chessboard(x, y, chessboard2, self.color)
        result = Node(self, chessboard2, -self.color, x, y)
        self.children[(x, y)] = result
        return result


@nb.jit(nopython=True)
def default_policy(chessboard, color1):
    random.seed(10)
    chessboard = chessboard.copy()
    color = color1
    while True:
        moves = next_moves(chessboard, color, True)
        if len(moves) == 0 and len(next_moves(chessboard, -color, False)) == 0:
            break
        if len(moves) == 0:
            color = -color
            moves = next_moves(chessboard, color, True)
        rand_index = random.randint(0, len(moves) - 1)
        x, y= moves[rand_index]
        # x,y = greedy(moves)
        chessboard = update_chessboard(x, y, chessboard, color)
        color = -color
    return who_win(chessboard, color1)


# @nb.jit(nopython=True)
def greedy(moves: []):
    result = (-1, -1)
    max_grade = -100
    for x, y in moves:
        x_distance = abs(x - 3.5)
        y_distance = abs(y - 3.5)
        para = x_distance + y_distance
        if x_distance == 3.5 and y_distance == 3.5:
            para = para * 20
        else:
            if x_distance == 3.5 and y_distance != 2.5:
                para = para + 3.5
            if y_distance == 3.5 and x_distance != 2.5:
                para = para + 3.5
            if x_distance == 3.5 and y_distance == 2.5:
                para = para - 2
            if y_distance == 3.5 and x_distance == 2.5:
                para = para - 2
            if x_distance == 2.5 and y_distance == 2.5:
                para = para / 1.5
        if para > max_grade:
            max_grade = para
            result = (x, y)
    return result

def is_terminal(chessboard):
    list1 = next_moves(chessboard, 1, False)
    list2 = next_moves(chessboard, -1, False)
    return len(list1) + len(list2) == 0


@nb.jit(nopython=True)
def next_moves(chessboard, color, flag):
    idx = np.where(chessboard == COLOR_NONE)
    idx_list = list(zip(idx[0], idx[1]))
    result = []
    for x, y in idx_list:
        c, p1, p2 = valid_position(chessboard, x, y, color)
        if c > 0:
                result.append((x, y))
    return result


def tree_policy(node: Node,copy):

    while not is_terminal(node.chessboard):
        # skip case
        if is_expanded(node):
            node = best_child(node)
        else:
            '''
            make sure to expand a new subNode
            '''
            moves = next_moves(node.chessboard, node.color, False)
            children = node.children
            for x, y in moves:
                if (x, y) not in children.keys():
                    return node.expand(x, y)
    return node


def backup(node: Node, reward, rootNode: Node):
    turn = 1
    global max_access, best_p
    while node is not None:
        node.access += 1
        if node.parent == rootNode and node.access > max_access:
            max_access = node.access
            best_p = (node.x, node.y)
        node.reward += reward * turn
        turn = turn * -1
        node = node.parent


def is_expanded(node: Node):
    moves = next_moves(node.chessboard, node.color, False)
    children = node.children

    return len(moves) == len(children)


# @nb.jit(nopython=True)
def best_move(node):
    best_node = None
    max_access = -1
    for sub_node in node.children.values():
        if sub_node.access > max_access:
            max_access = sub_node.access
            best_node = sub_node
    return best_node


def best_child(node: Node):
    best_node = None
    best_score = -1e9
    for sub_node in node.children.values():
        left = sub_node.reward / sub_node.access
        right = 1 / np.sqrt(2) * np.sqrt(np.log(node.access) / sub_node.access)
        score = left + right
        if score > best_score:
            best_node = sub_node
            best_score = score
    if best_node is None :
        cs = node.chessboard.copy()
        best_node = Node(parent=node,chessboard=cs,color=-node.color,x=-2,y=-2)
    return best_node


@nb.jit(nopython=True)
def who_win(chessboard: np.ndarray, color):
    idx_1 = np.where(chessboard == COLOR_WHITE)
    c_white = len(idx_1[0])
    idx_1 = np.where(chessboard == COLOR_BLACK)
    c_black = len(idx_1[0])
    result = (c_white - c_black) * color
    if result > 0:
        # current color wins
        return 1
    elif result < 0:
        return -1
    else:
        # 平局
        return 0


@nb.jit(nopython=True)
def update_chessboard(x, y, chessboard, color):
    # |||方向
    t = 1
    while x + t <= 7 and chessboard[x + t, y] == -color:
        t += 1
    if x + t <= 7 and chessboard[x + t, y] == color:
        chessboard[x:x + t + 1, y] = color

    t = 1
    while x - t >= 0 and chessboard[x - t, y] == -color:
        t += 1
    if x - t >= 0 and chessboard[x - t, y] == color:
        chessboard[x - t:x + 1, y] = color

    # ----------方向
    t = 1
    while y + t <= 7 and chessboard[x, y + t] == -color:
        t += 1
    if y + t <= 7 and chessboard[x, y + t] == color:
        chessboard[x, y:y + t + 1] = color

    t = 1
    while y - t >= 0 and chessboard[x, y - t] == -color:
        t += 1
    if y - t >= 0 and chessboard[x, y - t] == color:
        chessboard[x, y - t:y + 1] = color

    # \\\\方向
    t = 1
    while y + t <= 7 and x + t <= 7 and chessboard[x + t, y + t] == -color:
        t += 1
    if y + t <= 7 and x + t <= 7 and chessboard[x + t, y + t] == color:
        for i in range(t + 1):
            chessboard[x + i, y + i] = color

    t = 1
    while y - t >= 0 and x - t >= 0 and chessboard[x - t, y - t] == -color:
        t += 1
    if y - t >= 0 and x - t >= 0 and chessboard[x - t, y - t] == color:
        for i in range(t + 1):
            chessboard[x - i, y - i] = color

    # ///方向
    t = 1
    while y + t <= 7 and x - t >= 0 and chessboard[x - t, y + t] == -color:
        t += 1
    if y + t <= 7 and x - t >= 0 and chessboard[x - t, y + t] == color:
        for i in range(t + 1):
            chessboard[x - i, y + i] = color

    t = 1
    while y - t >= 0 and x + t <= 7 and chessboard[x + t, y - t] == -color:
        t += 1
    if y - t >= 0 and x + t <= 7 and chessboard[x + t, y - t] == color:
        for i in range(t + 1):
            chessboard[x + i, y - i] = color

    return chessboard
