import os
import json


def save_new_index(components_links, components_nodes, components_active_nodes, root_dir, coarsen_edge_weight):
	print('saving new index components...')
	for i in range(len(components_links)):
		root_dir_component = root_dir + 'cc/'
		comp = components_links[i]
		with open(root_dir_component + 'newInx_links.txt', 'w') as f:
			f.write('source' + '\t' + 'target' + '\n')
			for line in comp:
				f.write(str(line[0]) + '\t' + str(line[1]) + '\n')
		f.close()

	for i in range(len(components_nodes)):
		root_dir_component = root_dir + 'cc/'
		comp = components_nodes[i]
		with open(root_dir_component + 'newInx_nodes.txt', 'w') as f:
			f.write('node' + '\n')
			for line in comp:
				f.write(str(line) + '\n')
		f.close()

	for i in range(len(components_active_nodes)):
		root_dir_component = root_dir + 'cc/'
		comp = components_active_nodes[i]
		with open(root_dir_component + 'newInx_active_nodes.txt', 'w') as f:
			f.write('node' + '\t' + 'time' + '\n')
			for line in comp:
				f.write(str(line[0]) + '\t' + str(line[1]) + '\n')
		f.close()
	print('saving coarsening components...')
	for i in range(len(components_links)):
		root_dir_component = root_dir + 'cc/'
		if not os.path.exists(root_dir_component):
			os.makedirs(root_dir_component )
		comp = components_links[i]
		with open(root_dir_component + '/coarse.txt', 'w') as f:
			for line in comp:
				f.write(str(line[0]) + '\t' + str(line[1]) + '\t' + str(coarsen_edge_weight) + '\t' + str(coarsen_edge_weight) + '\n')
		f.close()


def new_indexing(components_links, components_nodes, components_active_nodes, coarsen_edge_weight, root_dir):
	D = {}
	# print('change nodes ids...')
	# print('components_nodescomponents_nodes:' + str(len(components_nodes)))
	for i in range(len(components_nodes)):
		comp_nodes = components_nodes[i]
		sorted_nodes = sorted(comp_nodes)
		# print('sorted_nodes: ' + str(len(sorted_nodes)))
		for j in range(len(sorted_nodes)):
			compnode = sorted_nodes[j]
			index = j + 1
			D[compnode] = {}
			D[compnode]['index'] = index
		# print('put comp id in dictionary...')
		for j in range(len(comp_nodes)):
			D[comp_nodes[j]]['comp'] = i
			comp_nodes[j] = D[comp_nodes[j]]['index']

		components_nodes[i] = comp_nodes
		# print('comp_nodes: ' + str(len(comp_nodes)))
		# print('changing link ids...')
		comp_links = components_links[i]
		for j in range(len(comp_links)):
			index1 = D[comp_links[j][0]]['index']
			index2 = D[comp_links[j][1]]['index']
			comp_links[j] = [index1, index2]
		components_links[i] = comp_links

		# print('change active nodes ids..')
		comp_active_nodes = components_active_nodes[i]
		for j in range(len(comp_active_nodes)):
			index = D[comp_active_nodes[j][0]]['index']
			comp_active_nodes[j][0] = index
		components_active_nodes[i] = comp_active_nodes

	# print('saving the dictionary file...')
	# JasonFile = open(root_dir + 'Index_Dictionary.json', 'wb')
	# json.dump(D, JasonFile, indent=4)
	# JasonFile.close()
	print('components_nodes: ' + str(len(components_nodes)))
	save_new_index(components_links, components_nodes, components_active_nodes, root_dir, coarsen_edge_weight)
	return components_links, components_nodes, components_active_nodes