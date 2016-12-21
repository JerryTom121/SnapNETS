__author__ = 'Sorour E.Amiri'

import graph_ALP as galp
import os
import networkx as nx
import sys
import subprocess
import creat_gephi_files2 as getgraph
import numpy as np
import time
import multiprocessing as mp
import os.path
import get_segment_feature as seg_feature
import feature_extraction as fe

mode_seg = 'all'
coarsen_edge_weight = '0.02'


def segmentation(args):
	all_segments, segment_index, components_active_nodes, i, root_dir, root_dir_component, root_dir_percent,  percent, score_dir, t_min, num_active, coarsening_mode, test_mode, mode, s_min, verbal = args

	segment = all_segments[segment_index]
	seg_score_dir, seg_active_dir, coarse_input_dir, coarse_output_dir, map_dir, temp_dir = get_comp_seg_dir(root_dir_percent,
																											 root_dir_component,
																											 i, segment)
	sub_active_node = get_sub_link_score(segment, components_active_nodes[i], score_dir, seg_score_dir, seg_active_dir, t_min)
	num_active[segment_index] += len(sub_active_node)
	start = time.time()
	coarse_net(coarsening_mode, coarse_input_dir, seg_score_dir, percent, coarse_output_dir, map_dir, temp_dir)
	elapsed = (time.time() - start)
	# print('end of coarse_net')

	comp_links_dir = root_dir_component + '/coarse.txt'  # 'newInx_links.txt'
	comp_nodes_dir = root_dir_component + 'newInx_nodes.txt'
	gephi_graph = root_dir_percent + 'graph' + str(i) + '_' + str(segment[0]) + '_' + str(segment[1]) + '.txt'
	gephi_nodes = root_dir_percent + 'nodes' + str(i) + '_' + str(segment[0]) + '_' + str(segment[1]) + '.txt'
	# print('get graph')
	coarse_link, coarse_node_label, coarse_link_weight, coarse_labels = getgraph.main(coarse_output_dir, map_dir,
																					  seg_active_dir, gephi_graph,
																					  gephi_nodes)
	# coarse_link, coarse_node_label, coarse_link_weight, coarse_labels = read_gephi(gephi_graph, gephi_nodes,
	# 																			   coarsening_mode, comp_links_dir,
	# 																			   comp_nodes_dir, seg_active_dir)
	# print('component_feature_time')
	component_feature_time = fe.extract_feature(coarse_link, coarse_node_label, coarse_link_weight, coarsening_mode)
	feature_dir = root_dir_percent + 'feature' + str(segment_index) + '.txt'
	with open(feature_dir, 'w') as f:
		for items in component_feature_time:
			f.write(str(items) + '\t')
	f.close()

	# print('parallel finished..')
	return 0


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


def read_features(all_segments, root_dir_percent, selected_features):
	component_feature = []
	act_num_coarsen = []
	for segment_index in range(len(all_segments)):
		feature_dir = root_dir_percent + '/feature' + str(segment_index) + '.txt'
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
			act_num_coarsen.append(0)
			component_feature_time = temp[:]
			component_feature.append(component_feature_time)

	return component_feature, act_num_coarsen


def normalize_features(feature_matrix, normalizing):
	# print('normalize_features: ' + normalize_mode)
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


def read_comps(root_dir_component, num_of_comps):
	components_links = []
	for i in range(num_of_comps):
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
		comp = []
		with open(root_dir_component + 'nodes.txt') as f:
			lines = f.read().splitlines()
			for j in range(1, len(lines)):
				line = lines[j]
				comp.append(int(line))
		components_nodes.append(comp)

	components_active_nodes = []
	for i in range(num_of_comps):
		comp = []
		with open(root_dir_component + 'active_nodes.txt') as f:
			lines = f.read().splitlines()
			for j in range(1, len(lines)):
				line = lines[j]
				items = line.split('\t')
				comp.append([int(items[0]), float(items[1])])
		components_active_nodes.append(comp)
	return components_links, components_nodes, components_active_nodes


def read_new_index(root_dir_component, num_of_comps, coarsen_edge_weight):
	components_links = []
	for i in range(num_of_comps):
		comp = []
		with open(root_dir_component + 'newInx_links.txt') as f:
			lines = f.read().splitlines()
			for j in range(1, len(lines)):
				line = lines[j]
				items = line.split('\t')
				comp.append([int(items[0]), int(items[1])])
		components_links.append(comp)

	components_nodes = []
	for i in range(num_of_comps):
		comp = []
		with open(root_dir_component + 'newInx_nodes.txt') as f:
			lines = f.read().splitlines()
			for j in range(1, len(lines)):
				line = lines[j]
				comp.append(int(line))
		components_nodes.append(comp)

	components_active_nodes = []
	for i in range(num_of_comps):
		comp = []
		with open(root_dir_component + 'newInx_active_nodes.txt') as f:
			lines = f.read().splitlines()
			for j in range(1, len(lines)):
				line = lines[j]
				items = line.split('\t')
				comp.append([int(items[0]), float(items[1])])
		components_active_nodes.append(comp)
	# print('saving coarsening components...')
	for i in range(len(components_links)):
		if not os.path.exists(root_dir_component):
			os.makedirs(root_dir_component)
		comp = components_links[i]
		coarse_file = root_dir_component + '/coarse.txt'
		if not os.path.isfile(coarse_file):
			with open(coarse_file, 'w') as f:
				for line in comp:
					f.write(str(line[0]) + '\t' + str(line[1]) + '\t' + str(coarsen_edge_weight) + '\t' + str(
						coarsen_edge_weight) + '\n')
			f.close()
	return components_links, components_nodes, components_active_nodes


def get_comp_seg_dir(root_dir_percent, root_dir_component, i, segment):
	seg_score_dir = root_dir_component + 'score_seg_' + str(segment[0]) + '_' + str(segment[1]) + '.txt'
	seg_active_dir = root_dir_component + 'active_nodes_seg_' + str(segment[0]) + '_' + str(segment[1]) + '.txt'
	coarse_input_dir = root_dir_component + '/coarse.txt'
	coarse_output_dir = root_dir_percent + 'coarse_' + str(i) + '_seg_' + str(segment[0]) + '_' + str(segment[1]) + '.txt'
	map_dir = root_dir_percent + 'final_map_' + str(i) + '_seg_' + str(segment[0]) + '_' + str(segment[1]) + '.txt'
	temp_dir = root_dir_percent + 'time_' + str(i) + '_seg_' + str(segment[0]) + '_' + str(segment[1]) + '.txt'
	return seg_score_dir, seg_active_dir, coarse_input_dir, coarse_output_dir, map_dir, temp_dir


def coarse_net(coarsening_mode, input_graph, score_file, percent, coarse_graph, map_file, temp):
	if coarsening_mode:
		subprocess.call(["src/CoarseNet", input_graph, score_file, percent, '0', coarse_graph, map_file, temp])
	else:
		print 'without Coarsening'


def save_sub_link_scores(sub_link_scores, sub_active_node, seg_score_dir, seg_active_dir):
	# print('saving segment component active nodes...')
	with open(seg_active_dir, 'w') as f:
		for item in sub_active_node:
			f.write(str(item) + '\n')
	f.close()
	# print('saving segment component scores...')
	with open(seg_score_dir, 'w') as f:
		for item in sub_link_scores:
			f.write(str(item[0]) + '\t' + str(item[1]) + '\t' + str(item[2]) + '\n')
	f.close()


def get_sub_link_score(segment, components_active_nodes, score_dir, seg_score_dir, seg_active_dir, t_min):
	# print('getting sub nodes...')
	sub_active_node = []
	D = {}
	for node, time in components_active_nodes:
		if (time < segment[1]) and (t_min <= time):
			sub_active_node.append(node)
			D[str(node)] = 0
	# print('remove candidates...')
	sub_link_scores = []
	with open(score_dir) as f:
		lines = f.read().splitlines()
	f.close()
	# print('close file:' + str(segment))
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


def get_link_score(component_dir, matlab_path, score_file):
	# print 'Scoring started'
	matlab_function = "getEigenScore('" + component_dir + "', '" + score_file + "')"
	subprocess.call([matlab_path, "-nosplash", "-nodisplay", "-r", matlab_function])
	# print 'Finished Scoring'


def get_all_segments(t_min, t_max, s_min, s_max, epsilon, seg_dir, data, verbal):
	error = 1e-5
	D = {}
	all_segments = []
	smallest_segments = []
	seg_size = s_min
	while seg_size <= s_max:
		y_s = t_min
		y_e = y_s + seg_size
		while y_s <= (t_max - epsilon):
			try:
				temp = D[str(y_s) + '-' + str(y_e)]
			except KeyError:
				D[str(y_s) + '-' + str(y_e)] = 0
				if reminder((y_e - y_s), s_min) <= error:
					all_segments.append([y_s, y_e])
				if seg_size == s_min:
					if reminder((y_e - y_s), s_min) <= error:
						smallest_segments.append([y_s, y_e])

			y_s += s_min
			y_e = min(y_s + seg_size, t_max)
		seg_size += s_min
	seg_size = s_max
	y_s = t_min
	y_e = y_s + seg_size
	while y_s <= (t_max - epsilon):
		try:
			temp = D[str(y_s) + '-' + str(y_e)]
		except:
			D[str(y_s) + '-' + str(y_e)] = 0
			if reminder((y_e - y_s), s_min) <= error:
				all_segments.append([y_s, y_e])
		y_s += s_min
		y_e = min(y_s + seg_size, t_max)
	i = 0
	if verbal:
		with open(seg_dir + 'all_segments.txt', 'w') as f:
			for line in all_segments:
				f.write(str(i) + ': ' + str(line[0]) + '\t' + str(line[1]) + '\n')
				i += 1

		with open(seg_dir + 'smallest_segments.txt', 'w') as f:
			for line in smallest_segments:
				f.write(str(line[0]) + '\t' + str(line[1]) + '\n')
	return all_segments, smallest_segments


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


def read_graph(link_dir, active_node_dir, cc_dir):
	active_nodes = []
	time = []
	links = []
	nodes = []
	node_dict = {}
	with open(link_dir) as f:
		lines = f.read().splitlines()
	lines.pop(0)
	counter = 0
	for line in lines:
		counter += 1
		items = line.split('\t')
		links.append([int(items[0]), int(items[1])])
		node_dict[int(items[0])] = 0
		node_dict[int(items[1])] = 0

	for node in node_dict:
		nodes.append(node)

	with open(active_node_dir) as f:
		lines = f.read().splitlines()
	lines.pop(0)
	for line in lines:
		items = line.split('\t')
		time.append(float(items[1]))
		active_nodes.append([int(float(items[0])), float(items[1])])
	t_min = min(time)
	epsilon = 0.1
	t_max = max(time) + epsilon

	coarse_file = cc_dir + 'coarse.txt'
	with open(coarse_file, 'w') as f:
		for line in links:
				f.write(str(line[0]) + '\t' + str(line[1]) + '\t' + str(coarsen_edge_weight) + '\t' +
						str(coarsen_edge_weight) + '\n')
	f.close()
	return [links], [nodes], [active_nodes], t_min, t_max, epsilon


def get_direction(root_dir, percent):
	# print('getting directories...')
	root_dir_percent = root_dir + '/' + percent + '/'
	root_dir_percent = root_dir_percent.replace('//', '/')
	active_node_dir = root_dir + '/infection.txt'
	active_node_dir = active_node_dir.replace('//', '/')
	link_dir = root_dir + '/graph.txt'
	link_dir = link_dir.replace('//', '/')
	node_dir = root_dir + '/nodes.txt'
	node_dir = node_dir.replace('//', '/')
	coarse_link_dir = root_dir + '/coarse_graph.txt'
	coarse_link_dir = coarse_link_dir.replace('//', '/')
	cc_dir = root_dir + '/cc/'
	cc_dir = cc_dir.replace('//', '/')

	if not os.path.exists(root_dir_percent):
		os.makedirs(root_dir_percent)

	if not os.path.exists(cc_dir):
		os.makedirs(cc_dir)
	# if not os.path.exists(root_dir_percent + str(coarsen_edge_weight) + '/'):
	# 	os.makedirs(root_dir_percent + str(coarsen_edge_weight) + '/')

	return link_dir, node_dir, active_node_dir, coarse_link_dir, root_dir_percent, cc_dir, root_dir


def main(data, percent, mode, thread, matlab_path):
	# print(normalize_mode)

	coarsen_edge_weight = '0.02'
	normalizing = True
	verbal = False

	# mode_seg = 'all'
	selected_features = range(8)

	# seg_feature_mode = 4
	test_mode = True
	coarsening_mode = True
	distance_mode = 'unweighted'
	# normal = 'raw'

	input_smin = '1' # The minimum length of a segment
	input_smax = 'null'

	link_dir, node_dir, active_node_dir, coarse_link_dir, root_dir_percent, cc_dir, root_dir = get_direction(data, percent)

	components_links, components_nodes, components_active_nodes, t_min, t_max, epsilon = read_graph(link_dir,
																									active_node_dir,
																									cc_dir)
	s_min = float(input_smin)
	num_of_comps = 1
	# if mode == '1':
	# links, nodes, active_nodes, t_min, t_max, epsilon = read_graph(link_dir, active_node_dir)
	# components_links, components_nodes, components_active_nodes = get_components(links, nodes, active_nodes, root_dir)
	if input_smax == 'null':
		s_max = t_max - t_min
	else:
		s_max = float(input_smax)

	seg_dir = root_dir + 's_min-' + str(s_min) + '-'
	all_segments, smallest_segments = get_all_segments(t_min, t_max, s_min, s_max, epsilon, seg_dir, data, verbal)

	features = []
	component_size = []
	num_active = [0] * len(smallest_segments)

	data_dir = root_dir + percent + '/'
	if not os.path.exists(data_dir):
		os.makedirs(data_dir)
	start1 = time.clock()
	start2 = time.time()

	component_dir = cc_dir + 'coarse.txt'
	score_dir = cc_dir + 'score.txt'
	get_link_score(component_dir, matlab_path, score_dir)
	component_size = len(components_nodes)
	inp = []
	################################
	if thread > 1:
		try:
			for segment_index in range(len(smallest_segments)):
				inp.append([smallest_segments, segment_index, components_active_nodes, 0, root_dir, cc_dir, root_dir_percent,
							percent, score_dir, t_min, num_active, coarsening_mode, test_mode, mode, s_min, verbal])

			print('multi cpu')
			pool = mp.Pool(thread)
			# print pool.map(segmentation, inp)
			pool.map(segmentation, inp)
			pool.close()
			pool.join()
		except KeyboardInterrupt:
			'terminate all...'
			pool.terminate()
	else:
		for segment_index in range(len(smallest_segments)):
			segmentation([smallest_segments, segment_index, components_active_nodes, 0, root_dir, cc_dir,
						  root_dir_percent, percent, score_dir, t_min, num_active, coarsening_mode, test_mode, mode,
						  s_min, verbal])

	###############################
	component_feature, act_num_coarsen = read_features(smallest_segments, root_dir_percent, selected_features)
	active_nodes_dir = cc_dir + 'active_nodes_seg_'
	num_active = read_num_active(smallest_segments, active_nodes_dir)
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
		feature_dir = root_dir + '/feature' + str(seg_index) + '_norm.txt'
		if verbal:
			with open(feature_dir, 'w') as f:
				for items in feature_matrix[seg_index]:
					f.write(str(items) + '\t')

	whole_feature_matrix, delta_dictionary, delta_array = seg_feature.main(feature_matrix, all_segments,
																		   smallest_segments, num_active)

	if verbal:
		for seg_index in range(len(all_segments)):
			feature_dir = root_dir_percent + 'feature' + str(seg_index) + '.txt'
			with open(feature_dir, 'w') as f:
				for items in whole_feature_matrix[seg_index]:
					f.write(str(items) + '\t')

	# ends here for storing the normalized feature values.
	maxp_len = len(smallest_segments) + 2
	ALP_arr = galp.main(whole_feature_matrix, all_segments, num_active, component_size, distance_mode, root_dir, maxp_len)

	end2 = time.time()
	end1 = time.clock()

	with open(root_dir + 'walltime.txt', 'w') as f:
		f.write(str(end2 - start2))
	with open(root_dir + 'clocktime.txt', 'w') as f:
		f.write(str(end1 - start1))


if __name__ == '__main__':
	data = sys.argv[1]
	matlab_path = sys.argv[2]
	# matlab_path = '/usr/local/R2011B/bin/matlab'
	thread = int(sys.argv[3])

	# data = './data/toy/' #sys.argv[1]
	# matlab_path = '/Applications/MATLAB_R2014a.app/bin/matlab' #sys.argv[2]
	# # matlab_path = '/usr/local/R2011B/bin/matlab'
	# thread = 1  # sys.argv[3]
	percent = '90' #sys.argv[4]
	mode = '1' #sys.argv[5]

	main(data, percent, mode, thread, matlab_path)
