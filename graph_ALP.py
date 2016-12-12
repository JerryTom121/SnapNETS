___author__ = 'Sorour Ekhtiari Amiri'
__email__ = 'esorour@vt.edu'

import numpy as np
import json
import sys
import os
import fast_ALP as falp


def Convert(ALP_arr, all_wins, FileName, FName):
	F = FileName + FName
	with open(F, 'wb')as f:
		# f.write(FileName + ':' + '\n')
		# f.write("['Source',")
		path = " "
		for i in ALP_arr:
			path += "'" + str(float(all_wins[int(i)][0])) + '-' + str(float(all_wins[int(i)][1])) + "'" + ' '
			f.write("'" + str(float(all_wins[int(i)][0])) + '-' + str(float(all_wins[int(i)][1])) + "'" + ',')
		# f.write("'Target']" + '\n')
		# path += " "
	f.close()
	print 'final segmentation: ' + path
	return path


def Finding_Neigbours(win_id, all_wins, node):
	Neigbours = []
	# YID.remove(node)
	E1 = all_wins[node][1]
	for item in win_id:
		B2 = all_wins[item][0]
		if E1 == B2:
			Neigbours.append(item)
	return Neigbours


def get_distance(node1, node2, wins_features, component_size, distance):
	# print('nodes1 is: ' + str(node1))
	a = np.array(wins_features[node1])
	# print('nodes2 is: ' + str(node2))
	b = np.array(wins_features[node2])
	if distance == 'weighted':
		feature_size = len(wins_features[0]) / len(component_size)
		base = 0
		for w in component_size:
			for index in range(feature_size):
				a[base + index] *= w
				b[base + index] *= w
			base += index + 1
	fraction = 1
	# fraction = abs(delta1 - delta2) / maxd
	dist = np.sqrt(np.sum((a - b) ** 2))
	return dist / fraction


def get_quality(Distance):
	quality = Distance
	return quality


def generate_graph(wins_features, all_wins, component_size, BeginTime, EndTime, distance, num_active):
	print 'generating segmentation graph...'
	G = {}
	N_points_arr = {}
	win_id = range(len(all_wins))
	Source = 'Source'
	Target = 'Target'
	G[Source] = []
	G[Target] = []
	TotalW = 0
	Totale = 0

	for node1 in win_id:
		size1 = all_wins[node1][1] - all_wins[node1][0]
		number1 = 0
		try:
			N_points_arr[size1].append(number1)
		except KeyError:
			N_points_arr[size1] = [number1]

		Neigbours = Finding_Neigbours(win_id, all_wins, node1)
		## Adding 2 extra nodes
		if (all_wins[node1][1] == EndTime) & (all_wins[node1][0] == BeginTime):
			continue
		if all_wins[node1][1] == EndTime:
			try:
				G[node1].append((Target, 0, 0, 0, 0))
			except KeyError:
				G[node1] = [(Target, 0, 0, 0, 0)]
		if all_wins[node1][0] == BeginTime:
			try:
				G[Source].append((node1, 0, 0, 0, 0))
			except KeyError:
				G[Source] = [(node1, 0, 0, 0, 0)]
		#################################################
		for node2 in Neigbours:
			size2 = all_wins[node2][1] - all_wins[node2][0]
			number2 = 0 #num_active[node2]
			if number1 <= number2:
				Min_n = number1
				Min_s = size1
			else:
				Min_n = number2
				Min_s = size2
			Distance = get_distance(node1, node2, wins_features, component_size, distance)
			quality = 0
			Weight = Distance
			TotalW += Weight
			Totale += 1
			try:
				G[node1].append((node2, Weight, Min_n, Min_s, quality))
			except KeyError:
				G[node1] = [(node2, Weight, Min_n, Min_s, quality)]
	Mean = {}
	Std = {}
	Graph = {}
	for Key in N_points_arr.keys():
		Mean[Key] = np.array(N_points_arr[Key]).mean()
		Std[Key] = np.array(N_points_arr[Key]).std()
	Total = sum(num_active)
	for Key in G.keys():
		for i in range(0, len(G[Key])):
			node2, Weight, Min_n, Min_s, quality = G[Key][i]
			if Min_s > 0:
				quality = get_quality(Weight)
				G[Key][i] = (node2, Weight, Min_n, Min_s, quality)
	for Key in G.keys():
		for node2, Weight, Min_n, Min_s, quality in G[Key]:
			try:
				Graph[Key][node2] = quality
			except KeyError:
				Graph[Key] = {}
				Graph[Key][node2] = quality
	Graph['Target'] = {}
	return Graph, G, (float(TotalW) / Totale)


def main(wins_features, all_wins, num_active, component_size, distance, data_dir, maxp_len):
	BeginTime = min(min(all_wins))
	EndTime = max(max(all_wins))

	G, G1, TotalAVG = generate_graph(wins_features, all_wins, component_size, BeginTime, EndTime, distance, num_active)
	l = maxp_len #len(all_wins)

	ALP_arr, avg, path_length = falp.main(G, 'Source', 'Target', l)
	FName = 'final_segmentation.txt'
	Name = 'ALP_segnum.txt'
	CostName = 'ALP-score.txt'

	Filename = data_dir
	if not os.path.exists(Filename):
		os.makedirs(Filename)
	path = Convert(ALP_arr, all_wins, Filename, FName)

	# with open(Filename + Name, 'wb')as f:
	# 	for j in ALP_arr:
	# 		f.write(str(j) + '\t')
	# with open(Filename + CostName, 'wb') as g:
	# 	g.write('AVG of path: ' + str(avg) + '\n')
	# 	g.write('Totla AVG: ' + str(TotalAVG))

	# JasonFile = open(Filename + 'Graph1.json', 'wb')
	# json.dump(G, JasonFile, indent=4)
	# JasonFile.close()
	return path


if __name__ == '__main__':
	wins_features = sys.argv[1]
	all_wins = sys.argv[2]
	num_active = sys.argv[3]
	component_size = sys.argv[4]
	distance = sys.argv[5]
	data_dir = sys.argv[6]
	maxp_len = sys.argv[7]
	main(wins_features, all_wins, num_active, component_size, distance, data_dir, maxp_len)
