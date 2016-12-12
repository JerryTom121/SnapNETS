import sys


def change_index(links1, links, nodes, infected):
	D = {}
	new_active = []
	new_node = []
	new_link =[]
	new_link1 = []
	# print('len node1: ' + str(len(nodes)))
	for j in range(len(nodes)):
		# compnode = nodes[j]
		# index = j
		D[nodes[j]] = {}
		D[nodes[j]]['index'] = j

	for j in range(len(nodes)):
		D[nodes[j]]['comp'] = 0
		new_node.append(D[nodes[j]]['index'])

	nodes = new_node[:]
	# print('changing link1 ids...')
	for j in range(len(links1)):
		index1 = D[links1[j][0]]['index']
		index2 = D[links1[j][1]]['index']
		# new_link1.append([D[links1[j][0]]['index'], D[links1[j][1]]['index'], links1[j][2]])
		links1[j][0] = D[links1[j][0]]['index']
		links1[j][1] = D[links1[j][1]]['index']
	# links1 = new_link1[:]
	# print('changing link ids...')
	for j in range(len(links)):
		index1 = D[links[j][0]]['index']
		index2 = D[links[j][1]]['index']
		# new_link.append([D[links[j][0]]['index'], D[links[j][1]]['index'], links[j][2]])
		links[j][0] = index1
		links[j][1] = index2

	# links = new_link[:]
	# print('change active nodes ids..')
	# print(str(active_nodes))
	for j in range(len(infected)):
		try:
			index = D[infected[j]]['index']
			new_active.append(index)
		except KeyError:
			a = 1
	infected = new_active[:]
	# print('infected: ' + str(max(infected)))
	# print('nodes: ' + str(max(nodes)))
	# print('nodes len: ' + str(len(nodes)))
	# print('new_active: ' + str(new_active)) # + '\t' + 'active: ' + str(active_nodes))
	# print('finished changing ids')
	return links1, links, nodes, infected


def main(graph_dir, nodes_dir, labels_dir, gephi_graph, gephi_nodes):
	edges = []
	edge1 = []
	weight = []

	D = {}
	with open(graph_dir) as f:
		lines = f.read().split('\n')
	f.close()
	for line in lines:
		items = line.split('\t')
		if len(items) == 3:
			try:
				I = str(items[0]) + '-' + str(items[1])
				temp1 = D[I]
			except:
				edges.append([items[0], items[1], 'Undirected', items[2]])
				edge1.append(items)
				weight.append(float(items[2]))
				I = str(items[0]) + '-' + str(items[1])
				D[I] = 0
				I = str(items[1]) + '-' + str(items[0])
				D[I] = 0

	nodes = []
	with open(nodes_dir) as f:
		lines = f.read().split('\n')
		for line in lines:
			items = line.split(' :')
			if items[0] != '':
				nodes.append(items[0])
	f.close()

	with open(labels_dir) as f:
		infected = []
		lines = f.read().split('\n')
		for line in lines:
			if line != '':
				infected.append(line)
	f.close()
	# print(str(infected))
	edge1, edges, nodes, infected = change_index(edge1, edges, nodes, infected)

	D = {}
	for item in infected:
		D[item] = 0

	node_label = []
	label = []
	for node in nodes:
		try:
			temp = D[node]
			node_label.append([node, 1])
			label.append(1)
		except:
			node_label.append([node, 0])
			label.append(0)

	with open(gephi_graph, 'w') as f:
		f.write('Source' + '\t' + 'Target' + '\t' + 'Type' + '\t' + 'Weight' + '\n')
		for items in edges:
			for item in items:
				f.write(str(item) + '\t')
			f.write('\n')
	f.close()

	with open(gephi_nodes, 'w') as f:
		f.write('Id' + '\t' + 'Label' + '\n')
		for line in node_label:
			for item in line:
				f.write(str(item) + '\t')
			f.write('\n')
	f.close()

	return edge1, node_label, weight, label

if __name__ == '__main__':
	graph_dir = sys.argv[1]
	nodes_dir = sys.argv[2]
	labels_dir = sys.argv[3]
	gephi_graph = sys.argv[4]
	gephi_nodes = sys.argv[5]

	main(graph_dir, nodes_dir, labels_dir, gephi_graph, gephi_nodes)