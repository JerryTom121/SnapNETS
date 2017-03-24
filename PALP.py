__author__ ='Sorour E.Amiri'

import sys
import operator
from multiprocessing import Pool, Value
import time

graph = {}
source = ''
target = ''
c = 2
level = 0
tree_nodes = ''
thread = 1
start_dict = {}
t_min = 0
t_max = 1
s_min = 1
seg2cpu = []
e_round = 10


def get_tree_levels(l, start_dict, t_min, t_max, s_min):
    tree_levels = {}
    if l == 0:
        tree_levels[source] = 0
    elif l == 1:
        for i in start_dict[t_min]:
            tree_levels[str(i)] = 0
    else:
        st = t_min + (l-1)*s_min
        while st < t_max:
            st = round(st, e_round)
            try:
                for key in start_dict[st]:
                    tree_levels[str(key)] = 0
            except KeyError:
                tmp = 0
            st += s_min
        tree_levels[target] = 0
    return tree_levels


def next_level(inp):
    crr_level_nodes, node2cpu = inp
    global tree_nodes
    global level
    next_level_tree_nodes = {}
    for nodes in node2cpu:
        if nodes in crr_level_nodes:
            nodes = str(nodes)
            try:
                pi, p_weight = tree_nodes[nodes][level]
            except KeyError:
                print nodes
                print level
                print crr_level_nodes
                print tree_nodes[nodes]
            for successor in graph[nodes]:
                weight = p_weight + graph[nodes][successor]
                try:
                    curr_pi, curr_weight = next_level_tree_nodes[successor]
                    if curr_weight < weight:
                        next_level_tree_nodes[successor] = (nodes, weight)
                except KeyError:
                    next_level_tree_nodes[successor] = (nodes, weight)
    try:
        t_pi, t_weight = tree_nodes[target][level]
        alp_list = t_weight / float(level - c)
    except(KeyError, ZeroDivisionError):
        alp_list = 0

    return alp_list, next_level_tree_nodes


def main(g, s, t, start_dict, t_min, t_max, s_min, trd, seg2cpu):
    global graph
    global source
    global target
    global c
    global level
    global tree_nodes
    global thread
    thread = trd
    graph = g
    source = s
    target = t
    alp_list = []
    tree_nodes = {}
    tree_levels = {}
    level = 0
    predecessor = 'null'
    weight = 0
    c = 2
    tree_nodes[source] = {}
    tree_nodes[source][level] = (predecessor, weight)
    # if thread > 1:
    #     pool = Pool(thread)

    start_time = time.time()

    if thread > 1:
        size2cpu = {}
        for i in range(len(seg2cpu)):
            for size in seg2cpu[i]:
                size = round(size, e_round)
                size /= s_min
                size = round(size, e_round)
                size2cpu[size] = i
        node2cpu = [[] for i in range(len(seg2cpu))]
        for key1 in start_dict:
            for key2 in start_dict[key1]:
                size = key2[1] - key2[0]
                size = round(size, e_round)
                n_seg = size/s_min
                n_seg = int(n_seg)
                if abs(n_seg*s_min - size) > 1e-10:
                    n_seg += 1

                node2cpu[size2cpu[n_seg]].append(str(key2))

        node2cpu[size2cpu[1]].append(source)
        node2cpu[size2cpu[2]].append(target)
    else:
        node2cpu = []
        for key1 in start_dict:
            for key2 in start_dict[key1]:
                node2cpu.append(str(key2))
        node2cpu.append(source)
        node2cpu.append(target)
    l = 0
    converged = False
    while not converged:
        # print l
        level = l
        tree_levels = get_tree_levels(level, start_dict, t_min, t_max, s_min)
        if thread > 1:
            ### parallel
            inp = []
            for i in range(len(node2cpu)):
                inp.append([tree_levels, node2cpu[i]])
            try:
                pool = Pool(thread)
                out = pool.map(next_level, inp)
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                'terminate all...'
                pool.terminate()
            v2, v3 = zip(*out)
            alp = 0
            tmp_crr_level = {}
            for key in xrange(len(v2)):
                if v2[key] > alp:
                    alp = v2[key]
                for n in v3[key]:
                    tmp_pi, tmp_w = v3[key][n]
                    try:
                        pi, w = tree_nodes[n][level + 1]
                        if w < tmp_w:
                            tree_nodes[n][level + 1] = (tmp_pi, tmp_w)
                    except KeyError:
                        try:
                            tree_nodes[n]
                        except KeyError:
                            tree_nodes[n] = {}

                        tree_nodes[n][level + 1] = (tmp_pi, tmp_w)

        else:
            ### sequential
            alp, next_level_tree_nodes = next_level([tree_levels, node2cpu])

            for key in next_level_tree_nodes:
                try:
                    tree_nodes[key][level + 1] = next_level_tree_nodes[key]
                except KeyError:
                    tree_nodes[key] = {}
                    tree_nodes[key][level + 1] = next_level_tree_nodes[key]
        alp_list.append(alp)
        l += 1
        if len(tree_levels) <= 1:
            if target in tree_levels:
                converged = True

    buf_alp_list = alp_list[:]
    buf_alp_list.sort()
    max_index, max_value = max(enumerate(alp_list), key=operator.itemgetter(1))

    end_time = time.time()
    # print 'time: ' + str(end_time - start_time) + ' s'
    try:
        pi, p_weight = tree_nodes[target][max_index]
        level = max_index - 1
        path = []
        while not level == 0:
            path.append(pi)
            pi, p_weight = tree_nodes[pi][level]
            level -= 1
        path.reverse()
    except KeyError:
        path = [0]
        max_value = 0
        max_index = 0

    # if thread > 1:
    #     pool.close()
    #     pool.join()
    return path, max_value, max_index - c

if __name__ == '__main__':
    # graph, source, target, max_length = generate_test_input()
    path, path_weight, path_length = main(graph, source, target, start_dict, t_min, t_max, s_min, thread, seg2cpu)
    print path, path_weight, path_length
