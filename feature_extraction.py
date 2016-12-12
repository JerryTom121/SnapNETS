import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as spl
import networkx as nx


def get_avg_deg_act_neighbor(G, active_set):
	avg_act_neighbor = []
	D = {}
	for node in active_set:
		D[node] = 1

	for node in active_set:
		act_neighbors = []
		neighbors = G.neighbors(node)
		for nodes in neighbors:
			try:
				temp = D[nodes]
				act_neighbors.append(nodes)
			except KeyError:
				temp = 0
		if len(act_neighbors) > 0:
			deg = list(G.degree(act_neighbors).values())
			avg_act_neighbor.append(np.average(deg))
		else:
			avg_act_neighbor.append(0)

	if len(avg_act_neighbor) > 0:
		Avg = np.average(avg_act_neighbor)
	else:
		Avg = 0
	return Avg


def get_active_pagerank(page_rank, active_set):
	f_vec = []
	for n in active_set:
		try:
			f_vec.append(page_rank[n])
		except KeyError:
			a = 1
	if len(f_vec) == 0:
		f_vec = 0

	f_vec = np.array(f_vec)
	return f_vec


def get_active_eigvec(eigvec, active_set):
	if len(active_set) > 0:
		f_vec = eigvec[active_set]
	else:
		f_vec = np.array([0])
	return f_vec


def get_lcc(G):
	component_list = list(nx.connected_component_subgraphs(G))
	lcc = component_list[0]
	# print('component_list: ' + str(component_list))
	return lcc


def get_star2(edge_list, active_set, inactive_set):
	# print('stars')
	star_act_core = 0
	star_inact_core = 0
	frontier_edge = 0
	nodes = {}
	# print('build dictionary..')
	# print('act')
	for i in active_set:
		nodes[i] = []
		nodes[i].append(1)
		nodes[i].append(0)
		nodes[i].append(0)
	# print('inact')
	for i in inactive_set:
		nodes[i] = []
		nodes[i].append(0)
		nodes[i].append(0)
		nodes[i].append(0)
	# print('get stars..')
	for node1, node2 in edge_list:
		label1 = nodes[node1][0]
		label2 = nodes[node2][0]
		if label1 == 1 and label2 == 0:
			frontier_edge += 1
			nodes[node1][1] += 1
			nodes[node2][2] += 1
		elif label1 == 0 and label2 == 1:
			frontier_edge += 1
			nodes[node1][2] += 1
			nodes[node2][1] += 1
	# print('get max stars')
	for key in nodes.keys():
		label, star1, star2 = nodes[key]
		if star1 > star_act_core:
			star_act_core = star1
		if star2 > star_inact_core:
			star_inact_core = star2

	if star_act_core < 3:
		star_act_core = 0
	if star_inact_core < 3:
		star_inact_core = 0
	return star_act_core, star_inact_core, frontier_edge


def get_min_max_adeg(deg_list, num):
	min_deg = 0
	max_deg = 0
	deg_list.sort()
	deg_list_len = len(deg_list)
	for i in range(num):
		min_deg += deg_list[i]
		max_deg += deg_list[deg_list_len - i - 1]

	if num > 0:
		max_deg /= float(num)
		min_deg /= float(num)
	else:
		max_deg = 1
		min_deg = 0
	return max_deg, min_deg


def calc_entropy2(text):
	import math
	log2 = lambda x: math.log(x) / math.log(2)
	exr = {}
	infoc = 0
	for each in text:
		try:
			exr[each] += 1
		except:
			exr[each] = 1
	textlen = len(text)
	items = 0
	for k, v in exr.items():
		items += 1
		freq = 1.0 * v / textlen
		infoc += freq * log2(freq)
	infoc *= -1
	return infoc, items


def calc_entropy3(p):
	import math
	log2 = lambda x: math.log(x) / math.log(2)
	infoc = 0
	for freq in p:
		# print('111')
		if freq == 0:
			print 'freq' + str(freq)
		infoc += freq * log2(freq)
		# print('222')
	infoc *= -1
	return infoc


def calc_entropy(labels):
	n_labels = len(labels)

	if n_labels <= 1:
		return 0

	counts = np.bincount(labels)
	probs = counts[np.nonzero(counts)] / float(n_labels)
	# print('probs: ' + str(probs) + '\t' + 'label: ' + str(labels))
	n_classes = len(probs)

	if n_classes <= 1:
		return 0
	return - np.sum(probs * np.log(probs)) / np.log(n_classes)


def get_star(valid_neighbor, node_set, G, m_deg):
	max_deg = 0
	best_node = ''
	best_n = []
	for node in node_set:
		neighbor_set = set(nx.neighbors(G, node))
		n_len = len(set(neighbor_set) & set(valid_neighbor))
		if max_deg < n_len:
			max_deg = 0 + n_len
			best_node = '' + node
			best_n = [] + list(neighbor_set)

	if max_deg <= 2:
		max_deg = 0
	# print('valid_neighbor: ' + str(valid_neighbor) + '\t' + 'best_n: ' + str(best_n) + '\t' + 'm_deg: ' + str(m_deg) + '\t' + 'f: ' + str(float(max_deg) / m_deg))
	return float(max_deg) / m_deg


def feature_raw(coarse_link, weights, G, active_set, inactive_set, coarsening_mode):
	# print('feature_raw')
	# lcc = get_lcc(G)
	feature_vector = []
	f_node_number = nx.number_of_nodes(G)
	# print('f_node_number: ' + str(f_node_number))
	if coarsening_mode:
		n = len(coarse_link)
		if n >= 1:
			# print('eig')
			c_link = np.array(coarse_link)
			c_link = c_link.astype(float)
			n = int(max(max(c_link[:, 0]), max(c_link[:, 1]))) + 1
			w_link = np.array([1.0] * len(weights))
			first_nodes = np.concatenate((c_link[:, 0], c_link[:, 1]), axis=0)
			second_nodes = np.concatenate((c_link[:, 1], c_link[:, 0]), axis=0)
			edge_weight = np.concatenate((w_link, w_link), axis=0)
			G_sparse = sp.coo_matrix((edge_weight, (first_nodes, second_nodes)), shape=(n, n))
			val, vec1 = spl.eigsh(G_sparse, k=1, which='LM', tol=1e-2)
			f_eig = np.absolute(val[0])
			# f_vec = get_active_eigvec(np.absolute(vec1[:, 0]), active_set)
			# average_vec = np.average(f_vec)
		else:
			f_eig = 0
			# average_vec = 0

		feature_vector.append(f_eig)
		# print 'edge'
		f_edge_number = float(nx.number_of_edges(G))
		feature_vector.append(f_edge_number)
		# print 'edge entropy'
		degree = list(nx.degree(G).values())
		m_deg = max(degree)
		# print 'calc'
		f_edge_entropy, num_w = calc_entropy2(weights)
		if num_w == 0:
			feature_vector.append(0)
		else:
			feature_vector.append(f_edge_entropy)
		# print 'avg cco'
		f_avg_cco = nx.average_clustering(G)
		feature_vector.append(f_avg_cco)

		# diameter = 0 #nx.diameter(lcc)
		# print('diameter')
		# feature_vector.append(diameter)
		# print('end diameter')
	else:
		for i in range(4):
			feature_vector.append(0)

		degree = list(nx.degree(G).values())
		m_deg = max(degree)

	epsilon = 1e-11
	f_infected = float(len(active_set))
	feature_vector.append(f_infected)

	page_rank = nx.pagerank(G, tol=1e-2)
	f_vec = get_active_pagerank(page_rank, active_set)
	# print ('f_vec' + str(f_vec))
	average_vec = np.average(f_vec)
	# total_vec = np.sum(f_vec)
	feature_vector.append(average_vec)
	# feature_vector.append(total_vec)
	# print '	print '
	active_deg = list(G.degree(active_set).values())
	if len(active_deg) > 0:
		f_avgd_infected = sum(active_deg) / float(len(active_deg))
	else:
		f_avgd_infected = 0
	max_adeg, min_adeg = get_min_max_adeg(degree, len(active_set))
	if max_adeg == min_adeg:
		min_adeg = 0

	if len(active_deg) > 0:
		if max_adeg - min_adeg > 0:
			feature_vector.append(f_avgd_infected)
		else:
			feature_vector.append(0)
	else:
		feature_vector.append(0)

	# print 'Avg'
	Avg = get_avg_deg_act_neighbor(G, active_set)
	feature_vector.append(Avg)
	inactive_deg = list(G.degree(inactive_set).values())
	if len(inactive_deg) > 0:
		f_entopy_deg_inactive, num_id = calc_entropy2(inactive_deg)
	else:
		f_entopy_deg_inactive = 0
		num_id = 1
	max_ient = calc_entropy3([1.0 / num_id] * num_id)
	if max_ient > epsilon:
		feature_vector.append(f_entopy_deg_inactive)
	else:
		feature_vector.append(0.0)
	# print 'star'
	star_act_core, star_inact_core, frontier_edge = get_star2(G.edges(), active_set, inactive_set)
	if m_deg > 0:
		feature_vector.append(star_act_core)
	else:
		feature_vector.append(0)

	return feature_vector


def feature_local(coarse_link, weights, G, active_set, inactive_set, coarsening_mode):
	feature_vector = []
	lcc = get_lcc(G)
	f_node_number = nx.number_of_nodes(G)
	if coarsening_mode:
		n = len(coarse_link)
		if n >= 1:
			c_link = np.array(coarse_link)
			c_link = c_link.astype(float)
			n = int(max(max(c_link[:, 0]), max(c_link[:, 1]))) + 1
			w_link = np.array([1.0] * len(weights))
			first_nodes = np.concatenate((c_link[:, 0], c_link[:, 1]), axis=0)
			second_nodes = np.concatenate((c_link[:, 1], c_link[:, 0]), axis=0)
			edge_weight = np.concatenate((w_link, w_link), axis=0)
			G_sparse = sp.coo_matrix((edge_weight, (first_nodes, second_nodes)), shape=(n, n))
			val, vec1 = spl.eigs(G_sparse, k=2, which='LM', tol=1e-2)
			f_eig = np.absolute(val[0])
			f_vec = np.absolute(vec1[0])
			average_vec = np.average(f_vec)
		else:
			f_eig = 0
			average_vec = 0

		min_eig = 0.0
		max_eig = f_node_number - 1
		feature_vector.append(((f_eig - min_eig) / (max_eig - min_eig)))
		max_vec = 1 / float(f_node_number)
		feature_vector.append(average_vec / max_vec)
		f_edge_number = float(nx.number_of_edges(G))
		min_edge_number = f_node_number - 1
		max_edge_number = f_node_number * (f_node_number - 1) * 0.5
		feature_vector.append(((f_edge_number - min_edge_number) / (max_edge_number - min_edge_number)))
		degree = list(nx.degree(G).values())
		m_deg = max(degree)

		f_edge_entropy, num_w = calc_entropy2(weights)
		if num_w == 0:
			feature_vector.append(0)
		else:
			min_went = 0
			max_went = calc_entropy3([1.0 / num_w] * num_w)
			feature_vector.append(((f_edge_entropy - min_went) / (max_went - min_went)))
		f_avg_cco = nx.average_clustering(G)
		min_cco = 0
		max_cco = 1
		feature_vector.append((f_avg_cco - min_cco) / (max_cco - min_cco))

		diameter = nx.diameter(lcc)
		feature_vector.append(diameter / float(f_node_number))
	else:
		for i in range(6):
			feature_vector.append(0)

		degree = list(nx.degree(G).values())
		m_deg = max(degree)

	epsilon = 1e-11
	f_infected = float(len(active_set))
	feature_vector.append((f_infected / f_node_number))

	active_deg = list(G.degree(active_set).values())
	if len(active_deg) > 0:
		f_avgd_infected = sum(active_deg) / float(len(active_deg))
	else:
		f_avgd_infected = 0
	max_adeg, min_adeg = get_min_max_adeg(degree, len(active_set))
	if max_adeg == min_adeg:
		min_adeg = 0

	if len(active_deg) > 0:
		if max_adeg - min_adeg > 0:
			feature_vector.append(((f_avgd_infected - min_adeg) / (max_adeg - min_adeg)))
		else:
			feature_vector.append(0)
	else:
		feature_vector.append(0)

	inactive_deg = list(G.degree(inactive_set).values())
	if len(inactive_deg) > 0:
		f_entopy_deg_inactive, num_id = calc_entropy2(inactive_deg)
	else:
		f_entopy_deg_inactive = 0
		num_id = 1
	max_ient = calc_entropy3([1.0 / num_id] * num_id)
	if max_ient > epsilon:
		feature_vector.append(f_entopy_deg_inactive / max_ient)
	else:
		feature_vector.append(0.0)
	star_act_core, star_inact_core, frontier_edge = get_star2(G.edges(), active_set, inactive_set)
	if m_deg > 0:
		feature_vector.append(star_act_core / float(m_deg))
	else:
		feature_vector.append(0)
	# feature_vector.append(star_inact_core / float(m_deg))
	# print('len(feature_vector): ' + str(len(feature_vector)))
	return feature_vector


def extract_feature(coarse_link, coarse_node_label, coarse_link_weight, coarse_labels, test_mode, coarsening_mode, normal):
	# print('extracting features...')
	links = coarse_link[:]
	# print('links: ' + str(len(links)))
	weights = coarse_link_weight[:]
	# print('weights: ' + str(len(weights)))
	nodes_labels = coarse_node_label[:]
	# print('nodes_labels: ' + str(len(nodes_labels)))

	G = nx.Graph()
	active_set = []
	inactive_set = []

	for node, label in nodes_labels:
		G.add_node(node, label=label)
		if label == 1:
			active_set.append(node)
		else:
			inactive_set.append(node)
	tuple = []
	for node1, node2, w in links:
		tuple.append((node1, node2, float(w)))

	G.add_weighted_edges_from(tuple)
	# G.add_edge(node1, node2, weight=float(w))

	if normal == 'local':
		feature_vector = feature_local(coarse_link, weights, G, active_set, inactive_set, coarsening_mode)
	elif normal == 'raw':
		feature_vector = feature_raw(coarse_link, weights, G, active_set, inactive_set, coarsening_mode)
	return feature_vector