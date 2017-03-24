__author__ = 'Sorour E.Amiri'

import os
import networkx as nx
import sys
import subprocess
import creat_gephi_files2 as getgraph
import numpy as np
import time
import os.path
import feature_extraction as fe
# import new_index as ni
import multiprocessing
import getEigenScore as gES
import PALP as falp


def segmentation(args):
    segment_index_arr, smallest_segments, root_dir, links, active_nodes, coarsening_mode, percent, \
    active_nodes_arr, min_n, max_n, G_size = args
    for segment_index in segment_index_arr:
        i = 0
        segment = smallest_segments[segment_index]
        components_active_nodes = active_nodes_arr[segment_index]
        root_dir_component = root_dir + percent + '/'
        score_dir = root_dir_component + 'score' + str(segment[0]) + '_' + str(segment[1]) + '.txt'

        remain_factor = round(1 - float(percent)/100, 2)
        if remain_factor * max_n < min_n:
            goal_szie = round(remain_factor * max_n)
        else:
            goal_szie = float(min_n)

        G_p = int((1 - (goal_szie / G_size[segment_index])) * 100)
        # print('G_p: ' + str(G_p))
        # print('goal_szie: ' + str(goal_szie))
        # print('G_size: ' + str(G_size[segment_index]))
        #
        seg_score_dir, seg_active_dir, coarse_input_dir, coarse_output_dir, map_dir, temp_dir = get_comp_seg_dir(root_dir_component,
                                                                                                                 i, segment)
        sub_active_node = get_sub_link_score(components_active_nodes, score_dir, seg_score_dir, seg_active_dir)
        start = time.time()
        coarse_net(coarsening_mode, coarse_input_dir, seg_score_dir, str(G_p), coarse_output_dir, map_dir, temp_dir)
        elapsed = (time.time() - start)
        # print('end of coarse_net')

        gephi_graph = root_dir + percent + '/graph' + str(i) + '_' + str(segment[0]) + '_' + str(segment[1]) + '.txt'
        gephi_nodes = root_dir + percent + '/nodes' + str(i) + '_' + str(segment[0]) + '_' + str(segment[1]) + '.txt'
        # print('get graph')
        coarse_link, coarse_node_label, coarse_link_weight, coarse_labels = getgraph.main(coarse_output_dir, map_dir,
                                                                                          seg_active_dir, gephi_graph,
                                                                                          gephi_nodes)
        # coarse_link, coarse_node_label, coarse_link_weight, coarse_labels = read_gephi(gephi_graph, gephi_nodes,
        # 																			   coarsening_mode, comp_links_dir,
        # 																			   comp_nodes_dir, seg_active_dir)
        # print('component_feature_time')
        component_feature_time = fe.extract_feature(coarse_link, coarse_node_label, coarse_link_weight, coarsening_mode)
        prefix = root_dir + percent + '/'
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        feature_dir = prefix + 'feature' + str(segment_index) + '.txt'
        with open(feature_dir, 'w') as f:
            for items in component_feature_time:
                f.write(str(items) + '\t')
        f.close()

        # print('parallel finished..')
    return 0


def get_dynamic_graphs(inp):
    links, active_nodes, segment_index_arr, root_dir_component, smallest_segments, root_dir = inp
    G_size = {}
    active_node_arr = {}
    max_n = None
    min_n = None
    for segment_index in segment_index_arr:
        index = 1
        segment = smallest_segments[segment_index]
        coarse_input_dir = root_dir_component + 'coarse' + str(segment[0]) + '_' + str(segment[1]) + '.txt'
        root_dir_component = root_dir + '/' + percent + '/'
        component = []
        score_dir = root_dir_component + 'score' + str(segment[0]) + '_' + str(segment[1]) + '.txt'
        #####get active nodes and links in each min_seg
        active_node_arr[segment_index] = []
        c_nodes = {}
        for key in links:
            link = links[key]
            if (link[2] == None) and (link[3] == None):
                component.append([link[0], link[1], 0.02, 0.02])
                for nn in [link[0], link[1]]:
                    try:
                        c_nodes[nn]
                    except KeyError:
                        c_nodes[nn] = index
                        index += 1
            elif (link[2] == None) and (link[3] >= segment[1]):
                component.append([link[0], link[1], 0.02, 0.02])
                for nn in [link[0], link[1]]:
                    try:
                        c_nodes[nn]
                    except KeyError:
                        c_nodes[nn] = index
                        index += 1
            elif (link[2] <= segment[1]) and (link[3] ==  None):
                component.append([link[0], link[1], 0.02, 0.02])
                for nn in [link[0], link[1]]:
                    try:
                        c_nodes[nn]
                    except KeyError:
                        c_nodes[nn] = index
                        index += 1
            elif (link[2] <= segment[1]) and (link[3] >= segment[1]):
                component.append([link[0], link[1], 0.02, 0.02])
                for nn in [link[0], link[1]]:
                    try:
                        c_nodes[nn]
                    except KeyError:
                        c_nodes[nn] = index
                        index += 1
        tmp_active = {}
        if segment[0] in active_nodes:
            for a_node in active_nodes[segment[0]]:
                if a_node in c_nodes:
                    active_node_arr[segment_index].append(c_nodes[a_node])
                    tmp_active[c_nodes[a_node]] = 0

        with open(coarse_input_dir, 'w') as f:
            for ii in range(len(component)):
                n1, n2, t1, t2 = component[ii]
                component[ii] = [c_nodes[n1], c_nodes[n2], t1, t2]
                n1, n2, t1, t2 = component[ii]
                f.write(str(n1) + '\t' + str(n2) + '\t' + str(t1) + '\t' + str(t2) + '\n')

        gES.main(component, score_dir, tol='1e-10')
        ###update score files
        with open(score_dir) as f:
            lines = f.readlines()
        with open(score_dir, 'w') as f:
            for line in lines:
                items = line.split('\t')
                n1 = int(items[0])
                n2 = int(items[1])
                s = float(items[2])
                both_active = (n1 in tmp_active) and (n2 in tmp_active)
                both_inactive = (not (n1 in tmp_active)) and (not (n2 in tmp_active))
                if both_active or both_inactive:
                    f.write(str(n1) + '\t' + str(n2) + '\t' + str(s) + '\n')
        ###
        if max_n == None:
            max_n = len(c_nodes)
        if min_n == None:
            min_n = len(c_nodes)
        if len(c_nodes) > max_n:
            max_n = len(c_nodes)
        if len(c_nodes) < min_n:
            min_n = len(c_nodes)
        G_size[segment_index] = len(c_nodes)

    return G_size, min_n, max_n, active_node_arr


def reminder(a, b):
    q = float(a) / float(b)
    if abs(q - round(q)) < 1e-6:
        q = round(q)

    else:
        q = int(q)

    rem = a - (q * b)

    return rem


def read_num_active(smallest_segments, active_nodes_dir):
    num_active = []
    for segment in smallest_segments:
        with open(active_nodes_dir + str(segment[0]) + '_' + str(segment[1]) + '.txt') as f:
            lines = f.readlines()
            num_active.append(float(len(lines)))
    for i in range(len(num_active) - 1):
        for j in range(i + 1, len(num_active)):
            num_active[j] += num_active[i]

    with open(active_nodes_dir + 'num_active.txt', 'w') as f:
        for i in range(len(num_active)):
            f.write(str(i) + '\t' + str(num_active[i]) + '\n')
    return num_active


def read_gephi(gephi_graph, gephi_nodes, coarsening_mode, comp_links_dir, comp_nodes_dir, seg_active_dir):
    coarse_link = []
    coarse_node_label = []
    coarse_link_weight = []
    coarse_labels = []
    if coarsening_mode:
        with open(gephi_graph) as f:
            lines = f.readlines()
        for i in range(1, len(lines)):
            line = lines[i]
            items = line.strip().split('\t')
            if len(items) >= 4:
                coarse_link.append([items[0], items[1], float(items[3])])
                coarse_link_weight.append(float(items[3]))

        with open(gephi_nodes) as f:
            lines = f.readlines()
        for i in range(1, len(lines)):
            line = lines[i]
            items = line.strip().split('\t')
            if len(items) >= 2:
                coarse_node_label.append([items[0], int(items[1])])
                coarse_labels.append(int(items[1]))
    else:
        with open(comp_links_dir) as f:
            lines = f.readlines()
        for i in range(1, len(lines)):
            line = lines[i]
            items = line.strip().split('\t')
            if len(items) >= 4:
                coarse_link.append([items[0], items[1], float(items[3])])
                coarse_link_weight.append(float(items[3]))
        nodes = []
        with open(comp_nodes_dir) as f:
            lines = f.readlines()
        for i in range(1, len(lines)):
            line = lines[i]
            items = line.strip().split('\t')
            if len(items) >= 1:
                nodes.append(items[0])

        with open(seg_active_dir) as f:
            infected = f.read().split('\n')
        D = {}
        for item in infected:
            D[item] = 0

        for node in nodes:
            try:
                temp = D[node]
                coarse_node_label.append([node, 1])
                coarse_labels.append(1)
            except KeyError:
                coarse_node_label.append([node, 0])
                coarse_labels.append(0)

    return coarse_link, coarse_node_label, coarse_link_weight, coarse_labels


def read_features(smallest_segments, root_dir, percent, selected_features):
    component_feature = []
    for segment_index in range(len(smallest_segments)):
        feature_dir = root_dir + percent + '/feature' + str(segment_index) + '.txt'
        component_feature_time = []
        with open(feature_dir) as f:
            lines = f.read().splitlines()
            line = lines[0]
            items = line.split('\t')
            for item in items:
                if not item == '':
                    component_feature_time.append(float(item))
            temp = []

            for item in selected_features:
                temp.append(component_feature_time[item])

            # act_num_coarsen.append(component_feature_time[9])
            component_feature_time = temp[:]
            component_feature.append(component_feature_time)

    return component_feature


def normalize_features(feature_matrix, normalizing):
    new_wins_features = []
    win_size = len(feature_matrix[0])
    for win in feature_matrix:
        for row in win:
            new_wins_features.append(row)
    new_wins_features = np.array(new_wins_features)
    index = len(new_wins_features[:, 0]) - 1
    temp_Wins_feature = new_wins_features[0:index, :]
    # if normal == 'global':
    if normalizing:
        min_array = temp_Wins_feature.min(0)
        max_array = temp_Wins_feature.max(0)
        temp = max_array - min_array
        for i in range(len(temp)):
            if abs(temp[i]) > 0:
                new_wins_features[:, i] = abs(new_wins_features[:, i] - min_array[i]) / abs(temp[i])
            else:
                new_wins_features[:, i] = abs(new_wins_features[:, i] - min_array[i])

    wins_features = []
    i = 0
    for row in new_wins_features:
        if i % win_size == 0:
            if i > 0:
                wins_features.append(temp)
            temp = list(row)
        else:
            temp = temp + list(row)
        i += 1
    wins_features.append(temp)
    return wins_features


def read_comps(root_dir, num_of_comps):
    components_links = []
    for i in range(num_of_comps):
        root_dir_component = root_dir + 'cc/'
        comp = []
        with open(root_dir_component + 'links.txt') as f:
            lines = f.read().splitlines()
            for j in range(1, len(lines)):
                line = lines[j]
                items = line.split('\t')
                comp.append([int(items[0]), int(items[1])])
        components_links.append(comp)

    components_nodes = []
    for i in range(num_of_comps):
        root_dir_component = root_dir + 'cc/'
        comp = []
        with open(root_dir_component + 'nodes.txt') as f:
            lines = f.read().splitlines()
            for j in range(1, len(lines)):
                line = lines[j]
                comp.append(int(line))
        components_nodes.append(comp)

    components_active_nodes = []
    for i in range(num_of_comps):
        root_dir_component = root_dir + 'cc/'
        comp = []
        with open(root_dir_component + 'active_nodes.txt') as f:
            lines = f.read().splitlines()
            for j in range(1, len(lines)):
                line = lines[j]
                items = line.split('\t')
                comp.append([int(items[0]), float(items[1])])
        components_active_nodes.append(comp)
    return components_links, components_nodes, components_active_nodes


def get_comp_seg_dir(root_dir_component, i, segment):
    seg_score_dir = root_dir_component + 'score_seg_' + str(segment[0]) + '_' + str(segment[1]) + '.txt'
    seg_active_dir = root_dir_component + 'active_nodes_seg_' + str(segment[0]) + '_' + str(segment[1]) + '.txt'
    coarse_input_dir = root_dir_component + 'coarse' + str(segment[0]) + '_' + str(segment[1]) + '.txt'
    coarse_output_dir = root_dir_component + 'coarse_' + str(i) + '_seg_' + str(segment[0]) + '_' + str(segment[1]) + '.txt'
    map_dir = root_dir_component + 'final_map_' + str(i) + '_seg_' + str(segment[0]) + '_' + str(segment[1]) + '.txt'
    temp_dir = root_dir_component + 'time_' + str(i) + '_seg_' + str(segment[0]) + '_' + str(segment[1]) + '.txt'
    return seg_score_dir, seg_active_dir, coarse_input_dir, coarse_output_dir, map_dir, temp_dir


def coarse_net(coarsening_mode, input_graph, score_file, percent, coarse_graph, map_file, temp):
    if coarsening_mode:
        subprocess.call(["src/CoarseNet", input_graph, score_file, percent, '0', coarse_graph, map_file, temp])
    else:
        print 'without Coarsening'


def save_sub_link_scores(sub_link_scores, sub_active_node, seg_score_dir, seg_active_dir):
    with open(seg_active_dir, 'w') as f:
        for item in sub_active_node:
            f.write(str(item) + '\n')
    f.close()
    with open(seg_score_dir, 'w') as f:
        for item in sub_link_scores:
            f.write(str(item[0]) + '\t' + str(item[1]) + '\t' + str(item[2]) + '\n')
    f.close()


def get_sub_link_score(sub_active_node, score_dir, seg_score_dir, seg_active_dir):
    D = {}
    sub_link_scores = []
    with open(score_dir) as f:
        lines = f.read().splitlines()
    f.close()
    for line in lines:
        nodes = line.split('\t')
        a = int(nodes[0])
        b = int(nodes[1])
        try:
            temp1 = D[str(a)]
        except:
            temp1 = 1
        try:
            temp2 = D[str(b)]
        except:
            temp2 = 1

        if (temp1 == 0) and (temp2 == 0):
            sub_link_scores.append(nodes)
        elif (temp1 == 1) and (temp2 == 1):
            sub_link_scores.append(nodes)
    save_sub_link_scores(sub_link_scores, sub_active_node, seg_score_dir, seg_active_dir)
    return sub_active_node


def get_link_score(component, score_file):
    # print 'Scoring started'
    # matlab_function = "getEigenScore('" + component_dir + "', '" + score_file + "')"
    # subprocess.call([matlab_path, "-nosplash", "-nodisplay", "-r", matlab_function])
    # component = []
    # with open(component_dir) as f:
    #     lines = f.readlines()
    # for line in lines:
    #     items = line.split('\t')
    #     component.append([int(items[0]), int(items[1]), float(items[2]), float(items[3])])
    link_score, lambda0 = gES.main(component, score_file, tol='1e-10')
    # print lambda0[0]
    # print 'Finished Scoring'


def get_smallest_segments(t_min, t_max, s_min, epsilon, verbal, seg_dir):
    error = 1e-5
    smallest_segments = []
    y_s = round(t_min, e_round)
    y_e = round(y_s + s_min, e_round)
    while y_s <= (t_max - epsilon):
        y_s = round(y_s, e_round)
        y_e = round(y_e, e_round)
        if reminder((y_e - y_s), s_min) <= error:
            smallest_segments.append([y_s, y_e])

        y_s += s_min
        y_e = min(y_s + s_min, t_max)
    ########################################################
    if verbal:
        with open(seg_dir + 'smallest_segments.txt', 'w') as f:
            for line in smallest_segments:
                f.write(str(line[0]) + '\t' + str(line[1]) + '\n')
    return smallest_segments


def save_components(components_links, components_nodes, components_active_nodes, root_dir):
    # print('saving components...')
    for i in range(len(components_links)):
        root_dir_component = root_dir + 'cc/'
        if not os.path.exists(root_dir_component):
            os.makedirs(root_dir_component)
        comp = components_links[i]
        with open(root_dir_component + 'links.txt', 'w') as f:
            f.write('source' + '\t' + 'target' + '\n')
            for line in comp:
                f.write(str(line[0]) + '\t' + str(line[1]) + '\n')
        f.close()

    for i in range(len(components_nodes)):
        root_dir_component = root_dir + 'cc/'
        comp = components_nodes[i]
        with open(root_dir_component + 'nodes.txt', 'w') as f:
            f.write('node' + '\n')
            for line in comp:
                f.write(str(line) + '\n')
        f.close()

    for i in range(len(components_active_nodes)):
        root_dir_component = root_dir + 'cc/'
        comp = components_active_nodes[i]
        with open(root_dir_component + 'active_nodes.txt', 'w') as f:
            f.write('node' + '\t' + 'time' + '\n')
            for line in comp:
                f.write(str(line[0]) + '\t' + str(line[1]) + '\n')
        f.close()


def get_components(links, nodes, active_nodes, root_dir):
    # print('getting components...')
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(links)
    components_links = []
    components_nodes = []
    components_active_nodes = []
    temp_active_nodes = []
    temp_active_nodes.append(active_nodes)
    temp_active_nodes = temp_active_nodes[0]
    component_list = list(nx.connected_component_subgraphs(G))

    # print('getting links of components...')
    for component in [component_list[0]]:
        # print('len of component: ' + str(len(component)))
        M = nx.edges(component)
        # print('links...')
        comp_links = [list(m) for m in M]
        # print('len of comp_links: ' + str(len(comp_links)))
        # print('append links...')
        components_links.append(comp_links)
        # print('getting nodes...')
        comp_nodes = nx.nodes(component)
        # print('len of comp_nodes: ' + str(len(comp_nodes)))
        # print('append nodes...')
        components_nodes.append(comp_nodes)
        # print('getting active nodes')
        comp_act_nodes = []
        D = {}
        C = {}
        # print('dictionary...')
        for n in comp_nodes:
            C[str(n)] = 0

        for act_node in temp_active_nodes:
            try:
                temp = C[str(act_node[0])]
                comp_act_nodes.append(act_node)
                D[str(act_node)] = 0
            except KeyError:
                temp = 0

        components_active_nodes.append(comp_act_nodes)
        # print('remove observed active nodes...')
        temp1 = []
        for x in temp_active_nodes:
            try:
                temp2 = D[str(x)]

            except KeyError:
                temp1.append(x)
        temp_active_nodes = []
        temp_active_nodes.append(temp1)
        temp_active_nodes = temp_active_nodes[0]

    save_components(components_links, components_nodes, components_active_nodes, root_dir)
    return components_links, components_nodes, components_active_nodes


def read_graph(link_dir, active_node_dir):
    print 'read_graph'
    active_nodes = {}
    links = {}
    nodes = {}
    t_min = None
    t_max = None
    with open(link_dir) as f:
        lines = f.read().splitlines()
    lines.pop(0)
    for line in lines:
        items = line.split('\t')
        n1 = int(items[0])
        n2 = int(items[1])

        if items[2].isdigit() and items[3].isdigit():
            t1 = float(items[2])
            t2 = float(items[3])
            _mint = min(t1, t2)
            _maxt = max(t1, t2)

        elif items[2].isdigit():
            t1 = float(items[2])
            t2 = None
            _mint = t1
            _maxt = t1
        elif items[3].isdigit():
            t1 = None
            t2 = float(items[3])
            _mint = t2
            _maxt = t2
        else:
            t1 = None
            t2 = None
            _mint = None
            _maxt = None

        if (t_min == None) or ((_mint < t_min) and (not _mint == None)):
            t_min = _mint
        if (t_max == None) or ((_maxt > t_max) and (not _maxt == None)):
            t_max = _maxt

        links[str([n1, n2])] = [n1, n2, t1, t2]
        nodes[n1] = 0
        nodes[n2] = 0

    with open(active_node_dir) as f:
        lines = f.read().splitlines()
    lines.pop(0)
    for line in lines:
        items = line.split('\t')
        n = int(float(items.pop(0)))
        for _time in items:
            try:
                _time = round(float(_time), e_round)
                # print _time
                if t_min == None:
                    t_min = _time
                if t_max == None:
                    t_max = _time
                if t_min > _time:
                    t_min = _time
                if t_max < _time:
                    t_max = _time
                if _time in active_nodes:
                    active_nodes[_time].append(n)
                else:
                    active_nodes[_time] = [n]
            except ValueError:
                # print _time
                continue
    epsilon = 0.1
    t_max += epsilon

    print 'num of links: ' + str(len(links))
    print 'num of nodes: ' + str(len(nodes))
    print 'num of active_nodes: ' + str(len(active_nodes))
    return links, nodes, active_nodes, t_min, t_max, epsilon


def get_direction(root_dir, percent, test_mode):
    # print('getting directories...')
    root_dir_percent = root_dir + percent + '/'
    if not test_mode:
        root_dir_percent = '../' + root_dir_percent
        root_dir = '../' + root_dir
    active_node_dir = root_dir + 'active.txt'
    link_dir = root_dir + 'links.txt'
    coarse_link_dir = root_dir + 'coarse_graph.txt'

    # if not os.path.exists(root_dir_percent):
    # 	os.makedirs(root_dir_percent)

    # if not os.path.exists(root_dir_percent + str(coarsen_edge_weight) + '/'):
    # 	os.makedirs(root_dir_percent + str(coarsen_edge_weight) + '/')

    return link_dir, active_node_dir, coarse_link_dir, root_dir_percent, root_dir

#
# def GaS(inp):
# 	seg2cpu, t_min, t_max, s_min, s_max, epsilon = inp
# 	error = 1e-5
# 	D = {}
# 	# all_segments = []
# 	start_dict = {}
# 	size_dict = {}
# 	for seg_size in seg2cpu:
# 		size_dict[seg_size] = []
# 		y_s = t_min
# 		y_e = y_s + seg_size
# 		while y_s <= (t_max - epsilon):
# 			try:
# 				temp = D[str(y_s) + '-' + str(y_e)]
# 			except KeyError:
# 				D[str(y_s) + '-' + str(y_e)] = 0
# 				if reminder((y_e - y_s), s_min) <= error:
# 					# all_segments.append([y_s, y_e])
# 					size_dict[seg_size].append([y_s, y_e])
# 					if y_s in start_dict:
# 						start_dict[y_s].append([y_s, y_e])
# 					else:
# 						start_dict[y_s] = [[y_s, y_e]]
#
# 			y_s += s_min
# 			y_e = min(y_s + seg_size, t_max)
#
# 	return size_dict, start_dict
#
#
# def get_all_segments(t_min, t_max, s_min, s_max, epsilon, seg_dir, seg2cpu):
# 	if thread_alp > 1:
# 		print 'parallel seg'
# 		inp = []
# 		for i in range(len(seg2cpu)):
# 			inp.append([seg2cpu[i], t_min, t_max, s_min, s_max, epsilon])
# 		out = pool.map(GaS, inp)
# 		size_dict = {}
# 		start_dict = {}
# 		tmp_v1, v3 = zip(*out)
# 		for i in range(len(seg2cpu)):
# 			for key in tmp_v1[i]:
# 				size_dict[key] = tmp_v1[i][key]
# 			for key1 in v3[i]:
# 				if not (key1 in start_dict):
# 					start_dict[key1] = []
# 				for value in v3[i][key1]:
# 					start_dict[key1].append(value)
#
# 	else:
# 		print 'seq seg'
# 		size_dict, start_dict = GaS([seg2cpu, t_min, t_max, s_min, s_max, epsilon])
# 	return start_dict, size_dict


def get_Avg(feature_matrix, smallest_segments, t_min, t_max, s_max, epsilon, thread, seg2cpu):
    acc = 1e+5
    D = {}
    num = {}
    s_min = round(smallest_segments[0][1] - smallest_segments[0][0], e_round)
    # check if the i in feature_matrix is true
    for i in range(len(smallest_segments)):
        small_seg = smallest_segments[i]
        s1 = round(small_seg[1] * acc) / acc
        D[s1] = feature_matrix[i]

    if thread > 1:
        inp = []
        for i in range(len(seg2cpu)):
            inp.append([seg2cpu[i], acc, s_min, D, t_min, t_max, s_max, epsilon, e_round])
        out = pool.map(gfp, inp)
        v1, v2 = zip(*out)
        wfm = {}
        start_dict = {}
        for v in v1:
            for key in v:
                wfm[key] = v[key]
        for i in range(len(seg2cpu)):
            for key1 in v2[i]:
                if not (key1 in start_dict):
                    start_dict[key1] = []
                for value in v2[i][key1]:
                    start_dict[key1].append(value)
    else:
        wfm, start_dict = gfp([seg2cpu, acc, s_min, D, t_min, t_max, s_max, epsilon, e_round])
    # whole_feature_matrix = []
    # for segment in all_segments:
    # 	whole_feature_matrix.append(wfm[str(segment)])

    return wfm, start_dict


def gfp(inp):
    seg2cpu, acc, s_min, D, t_min, t_max, s_max, epsilon, e_round = inp
    ############################################
    error = 1e-5
    DD = {}
    # all_segments = []
    start_dict = {}
    size_dict = {}
    wfm = {}
    ii = -1
    for size in seg2cpu:
        size = round(size, e_round)
        size_dict[size] = []
        y_s = t_min
        y_e = y_s + size
        while y_s <= (t_max - epsilon):
            y_s = round(y_s, e_round)
            y_e = round(y_e, e_round)
            try:
                temp = DD[str(y_s) + '-' + str(y_e)]
            except KeyError:
                DD[str(y_s) + '-' + str(y_e)] = 0
                if reminder((y_e - y_s), s_min) <= error:
                    # all_segments.append([y_s, y_e])
                    size_dict[size].append([y_s, y_e])
                    if y_s in start_dict:
                        start_dict[y_s].append([y_s, y_e])
                    else:
                        start_dict[y_s] = [[y_s, y_e]]

            y_s += s_min
            y_e = min(y_s + size, t_max)
        ############################################
        for segment in size_dict[size]:
            ii += 1
            avg = []
            temp = round(segment[1], e_round)
            avg = avg + D[temp]
            t_num = (size / s_min)

            # for j in list(np.arange(segment[0] + s_min, segment[1], s_min)):
            j = segment[0] + s_min
            while j < segment[1]:
                j = round(j * acc) / acc
                feature_v = D[j]
                for i in range(len(feature_v)):
                    avg[i] += feature_v[i]
                j += s_min

            for i in range(len(avg)):
                avg[i] /= t_num
            wfm[str(segment)] = avg
    return wfm, start_dict


def generate_graph(start_dict, all_features):
    print 'generate_graph'
    Source = 'Source'
    Target = 'Target'
    Graph = {}
    Graph[Source] = {}
    Graph[Target] = {}
    for key in start_dict:
        for value in start_dict[key]:
            G_key = str(value)
            a = all_features[G_key]
            Len = len(a)
            Graph[G_key] = {}
            begin_time = value[0]
            end_time = value[1]
            if begin_time == t_min:
                Graph[Source][G_key] = 0
            if end_time > (t_max - s_min):
                Graph[G_key][Target] = 0
            if end_time in start_dict:
                for neighbor in start_dict[end_time]:
                    b = all_features[str(neighbor)]
                    dist = 0
                    for i in range(Len):
                        dist += (a[i] - b[i])**2
                    Graph[G_key][str(neighbor)] = (dist ** 0.5)
    return Graph
#############################################################
#############################################################
########################  Main  #############################
#############################################################
#############################################################
data = sys.argv[1] # './data/toy_dynamic/'
thread = int(sys.argv[2])
thread_alp = int(sys.argv[3])
tmp_smin = 1.0
percent = '90'
mode = '1'

mode_seg = 'all'
coarsen_edge_weight = '0.02'
verbal = False
test_mode = True
normalizing = True
## reserve threads
max_thread = max(thread, thread_alp)
if max_thread > 1:
    pool = multiprocessing.Pool(max_thread)

# mode_seg = 'all'
selected_features = range(8)

# seg_feature_mode = 4
coarsening_mode = True
distance_mode = 'unweighted'
# normal = 'raw'

input_smin = '1'
input_smax = 'null'
e_round = 10
link_dir, active_node_dir, coarse_link_dir, root_dir_percent, root_dir = get_direction(data, percent, test_mode)

links, nodes, active_nodes, t_min, t_max, epsilon = read_graph(link_dir, active_node_dir)
s_min = tmp_smin #float(input_smin)
epsilon = s_min/10.0
num_of_comps = 1
# if mode == '1':
# components_links, components_nodes, components_active_nodes = get_components(links, nodes, active_nodes, root_dir)
# components_links, components_nodes, components_active_nodes = ni.new_indexing(components_links, components_nodes,
#                                                                               components_active_nodes,
#                                                                               coarsen_edge_weight, root_dir)
# else:
# 	components_links, components_nodes, components_active_nodes = read_new_index(root_dir, num_of_comps,
# 																				 coarsen_edge_weight)
if input_smax == 'null':
    s_max = t_max - t_min
else:
    s_max = float(input_smax)
seg_dir = root_dir + 's_min-' + str(s_min) + '-'
if thread_alp > 1:
    seg2cpu = [[] for i in range(thread_alp)]
    s_min = round(s_min, e_round)
    s = s_min
    i = 0
    while s <= s_max:
        s = round(s, e_round)
        seg2cpu[i % thread_alp].append(s)
        s += s_min
        i += 1
    # seg2cpu[i % thread_alp].append(s)
else:
    seg2cpu = []
    s = s_min
    while s <= s_max:
        s = round(s, e_round)
        seg2cpu.append(s)
        s += s_min
    # seg2cpu.append(s)
s_total = time.time()
# s_get_all_segments = time.time()
smallest_segments = get_smallest_segments(t_min, t_max, s_min, epsilon, verbal, seg_dir)
print('num of snapshots: ' + str(len(smallest_segments)))

if thread > 1:
    snapshot2cpu = [[] for i in range(thread)]
    for i in range(len(smallest_segments)):
        snapshot2cpu[i % thread].append(i)

else:
    snapshot2cpu = range(len(smallest_segments))
features = []

data_dir = root_dir + percent + '/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


start1 = time.clock()
start2 = time.time()


################################
s_sumarry = time.time()
if thread > 1:
    inp = []
    for i in range(len(snapshot2cpu)):
        inp.append([links, active_nodes, snapshot2cpu[i], data_dir, smallest_segments, root_dir])
    out = pool.map(get_dynamic_graphs, inp)
    G_size = {}
    active_nodes_arr = {}
    v1, v2, v3, v4 = zip(*out)
    min_n = v2[0]
    max_n = v3[0]
    for i in range(len(snapshot2cpu)):
        for key in v1[i]:
            G_size[key] = v1[i][key]
            active_nodes_arr[key] = v4[i][key]
        if min_n == None:
            min_n = v2[i]
        elif min_n > v2[i]:
            min_n = v2[i]
        if max_n == None:
            max_n = v3[i]
        elif max_n < v3[i]:
            max_n = v3[i]
    inp = []
    for i in range(len(snapshot2cpu)):
        inp.append([snapshot2cpu[i], smallest_segments, root_dir, links, active_nodes, coarsening_mode, percent,
                    active_nodes_arr, min_n, max_n, G_size])
    start3 = time.time()
    print('multi cpu')
    # print pool.map(segmentation, inp)
    pool.map(segmentation, inp)
else:
    G_size, min_n, max_n, active_node_arr = get_dynamic_graphs([links, active_nodes, snapshot2cpu, data_dir, smallest_segments, root_dir])
    segmentation([snapshot2cpu, smallest_segments, root_dir, links, active_nodes, coarsening_mode, percent,
                  active_node_arr, min_n, max_n, G_size])

###############################
component_feature = read_features(smallest_segments, root_dir, percent, selected_features)
features.append(component_feature)

feature_matrix = []
for seg_index in range(len(features[0])):
    temp_seg = []
    for comp_index in range(len(features)):
        temp_seg.append(features[comp_index][seg_index])
    feature_matrix.append(temp_seg)
last_index = len(feature_matrix) - 1
feature_matrix = normalize_features(feature_matrix, normalizing)

for seg_index in range(len(features[0])):
    feature_dir = data_dir + '/feature' + str(seg_index) + '_norm.txt'
    if verbal:
        with open(feature_dir, 'w') as f:
            for items in feature_matrix[seg_index]:
                f.write(str(items) + '\t')

s_get_Avg = time.time()
all_features, start_dict = get_Avg(feature_matrix, smallest_segments, t_min, t_max, s_max, epsilon, thread_alp, seg2cpu)

feature_dir_perfix = data_dir + '/'
if not os.path.exists(feature_dir_perfix):
    os.makedirs(feature_dir_perfix)
if verbal:
    for seg in all_features:
        feature_dir = feature_dir_perfix + 'feature' + str(seg) + '.txt'
        with open(feature_dir, 'w') as f:
            for items in all_features[seg]:
                f.write(str(items) + '\t')

# ends here for storing the normalized feature values.
s_generate_graph = time.time()
Graph = generate_graph(start_dict, all_features)
s_alp = time.time()
ALP_arr, avg, path_length = falp.main(Graph, 'Source', 'Target', start_dict, t_min, t_max, s_min, 1, seg2cpu)
FName = root_dir + 'final_segmentation.txt'
end2 = time.time()
end1 = time.clock()
if max_thread > 1:
    pool.close()
    pool.join()

with open(root_dir + 'wall_time.txt', 'w') as f:
    f.write(str(end2 - start2))
with open(root_dir + 'clock_time.txt', 'w') as f:
    f.write(str(end1 - start1))
with open(FName, 'w') as f:
    for seg in ALP_arr:
        try:
            seg = seg.replace('[', '')
            seg = seg.replace(']', '')
            seg = seg.replace(',', '-')
            f.write(seg + '\t')
        except AttributeError:
            continue
print ALP_arr
