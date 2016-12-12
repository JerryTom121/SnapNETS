__author__ ='Sorour E.Amiri'

import sys
import numpy as np
import time


def get_diff_node(feature_matrix, all_segments, smallest_segments, num_active):
	whole_feature_matrix = []
	D = {}
	num = {}
	delta_array = []
	delta_dictionary = {}

	for i in range(len(smallest_segments)):
		small_seg = smallest_segments[i]
		D[small_seg[1]] = feature_matrix[i]
		num[small_seg[1]] = num_active[i]

	s_min = small_seg[1] - small_seg[0]
	# print('s_min: ' + str(s_min))

	ii = -1
	for segment in all_segments:
		ii += 1
		try:
			fj = D[segment[0]]
			exist = True
		except:
			exist = False
		if exist:
			# print('can be added')
			# print(str(segment))
			f_end = D[segment[1]]
			# delta_n = fi[act_index] - fj[act_index] #
			# delta_n = (segment[1] - segment[0]) / s_min
			# print('delta_n: ' + str(delta_n))
			# delta_n = int(delta_n)
			# delta_n = num[segment[1]] - num[segment[0]]
			# delta_n = 1
			# delta_array.append(delta_n)
			# delta_dictionary[str(segment)] = delta_n
			time0 = time.localtime(segment[0])
			time1 = time.localtime(segment[1])
			# print('delta_n segment' + str(ii) + ': ' + str(time0.tm_mday) + '-' + str(time1.tm_mday) + ': ' + str(delta_n))
			temp = []

			###############################
			avg = f_end
			# if delta_n == 0:
			# 	print('delta_n = 0')
			# 	for i in range(len(f_end)):
			# 		temp.append(0)
			# else:

			t_num = (segment[1] - segment[0]) / s_min
			for j in list(np.arange(segment[0], segment[1], s_min)):
				j = round(j* (1e+5))/1e+5
				# print('j:' + str(j))
				feature_v = D[j]
				for i in range(len(feature_v)):
					avg[i] += feature_v[i]

			for i in range(len(avg)):
				avg[i] = avg[i] / t_num
			#################################
			# deltaf = 0
			# for i in range(len(fi)):
				# temp.append((fi[i] - fj[i]) / delta_n)
				# temp.append((fi[i]) / delta_n)
				# temp.append((fi[i] - fj[i]))
				# temp.append(Avg / delta_n)
			for iii in range(delta_n + 1):
				point = segment[0] + (iii * s_min)
				middle_feature = D[point]
				temp.append(middle_feature)

				# deltaf += ((fi[i] - fj[i]) ** 2)
			whole_feature_matrix.append(temp)
			# print('deltaf segment' + str(ii) + ': ' + str(time0.tm_mday) + '-' + str(time1.tm_mday) + ': ' + str(deltaf ** 0.5))
		else:
			# print('can not be added')
			# print(str(segment[0]))
			# whole_feature_matrix.append(D[segment[1]])
			# fi = D[segment[1]]
			# temp = []
			# for i in range(len(fi)):
			# 	temp.append(0)
			# whole_feature_matrix.append(temp)
			# print(str(segment))
			# fi = D[segment[1]]
			# fj = feature_matrix[Len]
			# fj = feature_matrix[0]
			# temp1 = smallest_segments[0]
			# print('smallest_segments[0]: ' + str(smallest_segments[0]))
			# fj = D[temp1[1]]
			# delta_n = fi[act_index] #
			delta_n = (segment[1] - segment[0]) / s_min
			# print('delta_n: ' + str(delta_n))
			delta_n = int(delta_n)
			# delta_n = 1
			# delta_n = num[segment[1]]
			delta_array.append(delta_n)
			delta_dictionary[str(segment)] = delta_n

			# Avg = fi
			# t_num = ((segment[1] - segment[0]) / s_min) - 1
			# for j in list(np.arange(segment[0] + s_min, segment[1], s_min)):
			# 	feature_v = D[j]
			# 	for i in range(len(feature_v)):
			# 		Avg[i] += feature_v[i]
			#
			# for i in range(len(Avg)):
			# 	Avg[i] = Avg[i] / t_num

			temp = []
			deltaf = 0
			# for i in range(len(fi)):
				# temp.append((fi[i] - fj[i]) / delta_n)
				# temp.append((fi[i]) / delta_n)
				# temp.append((fi[i] - fj[i]))
				# temp.append(Avg / delta_n)
			for iii in range(1, delta_n + 1):
				point = segment[0] + (iii * s_min)
				middle_feature = D[point]
				temp.append(middle_feature)

				# deltaf += ((fi[i] - fj[i]) ** 2)
			whole_feature_matrix.append(temp)
			time0 = time.localtime(segment[0])
			time1 = time.localtime(segment[1])
			# print('delta_n segment' + str(ii) + ': ' + str(time0.tm_mday) + '-' + str(time1.tm_mday) + ': ' + str(delta_n))
			# print('deltaf segment' + str(ii) + ': ' + str(time0.tm_mday) + '-' + str(time1.tm_mday) + ': ' + str(deltaf ** 0.5))
	# print('whole_feature_matrix len: ' + str(len(whole_feature_matrix)))
	# print('whole_feature_matrix len: ' + str(len(whole_feature_matrix)))
	return whole_feature_matrix, delta_dictionary, delta_array


def get_node_based_derivative(feature_matrix, all_segments, smallest_segments, num_active):
	whole_feature_matrix = []
	D = {}
	num = {}
	delta_array = []
	delta_dictionary = {}

	for i in range(len(smallest_segments)):
		small_seg = smallest_segments[i]
		D[small_seg[1]] = feature_matrix[i]
		num[small_seg[1]] = num_active[i]

	s_min = small_seg[1] - small_seg[0]
	# print('s_min: ' + str(s_min))

	ii = -1
	for segment in all_segments:
		ii += 1
		try:
			fj = D[segment[0]]
			exist = True
		except:
			exist = False
		if exist:
			# print('can be added')
			# print(str(segment))
			f_end = D[segment[1]]
			# delta_n = fi[act_index] - fj[act_index] #
			# delta_n = (segment[1] - segment[0]) / s_min
			# print('delta_n: ' + str(delta_n))
			# delta_n = int(delta_n)
			# delta_n = num[segment[1]] - num[segment[0]]
			# delta_n = 1
			# delta_array.append(delta_n)
			# delta_dictionary[str(segment)] = delta_n
			time0 = time.localtime(segment[0])
			time1 = time.localtime(segment[1])
			# print('delta_n segment' + str(ii) + ': ' + str(time0.tm_mday) + '-' + str(time1.tm_mday) + ': ' + str(delta_n))
			temp = []

			###############################
			avg = f_end
			# if delta_n == 0:
			# 	print('delta_n = 0')
			# 	for i in range(len(f_end)):
			# 		temp.append(0)
			# else:

			t_num = (segment[1] - segment[0]) / s_min
			for j in list(np.arange(segment[0], segment[1], s_min)):
				feature_v = D[j]
				for i in range(len(feature_v)):
					avg[i] += feature_v[i]

			for i in range(len(avg)):
				avg[i] = avg[i] / t_num
			#################################
			# deltaf = 0
			# for i in range(len(fi)):
				# temp.append((fi[i] - fj[i]) / delta_n)
				# temp.append((fi[i]) / delta_n)
				# temp.append((fi[i] - fj[i]))
				# temp.append(Avg / delta_n)
			for iii in range(delta_n + 1):
				point = segment[0] + (iii * s_min)
				middle_feature = D[point]
				temp.append(middle_feature)

				# deltaf += ((fi[i] - fj[i]) ** 2)
			whole_feature_matrix.append(temp)
			# print('deltaf segment' + str(ii) + ': ' + str(time0.tm_mday) + '-' + str(time1.tm_mday) + ': ' + str(deltaf ** 0.5))
		else:
			# print('can not be added')
			# print(str(segment[0]))
			# whole_feature_matrix.append(D[segment[1]])
			# fi = D[segment[1]]
			# temp = []
			# for i in range(len(fi)):
			# 	temp.append(0)
			# whole_feature_matrix.append(temp)
			print(str(segment))
			# fi = D[segment[1]]
			# fj = feature_matrix[Len]
			# fj = feature_matrix[0]
			# temp1 = smallest_segments[0]
			# print('smallest_segments[0]: ' + str(smallest_segments[0]))
			# fj = D[temp1[1]]
			# delta_n = fi[act_index] #
			delta_n = (segment[1] - segment[0]) / s_min
			# print('delta_n: ' + str(delta_n))
			delta_n = int(delta_n)
			# delta_n = 1
			# delta_n = num[segment[1]]
			delta_array.append(delta_n)
			delta_dictionary[str(segment)] = delta_n

			# Avg = fi
			# t_num = ((segment[1] - segment[0]) / s_min) - 1
			# for j in list(np.arange(segment[0] + s_min, segment[1], s_min)):
			# 	feature_v = D[j]
			# 	for i in range(len(feature_v)):
			# 		Avg[i] += feature_v[i]
			#
			# for i in range(len(Avg)):
			# 	Avg[i] = Avg[i] / t_num

			temp = []
			deltaf = 0
			# for i in range(len(fi)):
				# temp.append((fi[i] - fj[i]) / delta_n)
				# temp.append((fi[i]) / delta_n)
				# temp.append((fi[i] - fj[i]))
				# temp.append(Avg / delta_n)
			for iii in range(1, delta_n + 1):
				point = segment[0] + (iii * s_min)
				middle_feature = D[point]
				temp.append(middle_feature)

				# deltaf += ((fi[i] - fj[i]) ** 2)
			whole_feature_matrix.append(temp)
			time0 = time.localtime(segment[0])
			time1 = time.localtime(segment[1])
			# print('delta_n segment' + str(ii) + ': ' + str(time0.tm_mday) + '-' + str(time1.tm_mday) + ': ' + str(delta_n))
			# print('deltaf segment' + str(ii) + ': ' + str(time0.tm_mday) + '-' + str(time1.tm_mday) + ': ' + str(deltaf ** 0.5))
	# print('whole_feature_matrix len: ' + str(len(whole_feature_matrix)))
	# print('whole_feature_matrix len: ' + str(len(whole_feature_matrix)))
	return whole_feature_matrix, delta_dictionary, delta_array


def get_last_derivative(feature_matrix, all_segments, smallest_segments, num_active):
	whole_feature_matrix = []
	D = {}
	num = {}
	delta_array = []
	delta_dictionary = {}

	for i in range(len(smallest_segments)):
		small_seg = smallest_segments[i]
		D[small_seg[1]] = feature_matrix[i]
		num[small_seg[1]] = num_active[i]

	s_min = small_seg[1] - small_seg[0]
	# print('s_min: ' + str(s_min))

	ii = -1
	for segment in all_segments:
		ii += 1
		try:
			fj = D[segment[0]]
			exist = True
		except:
			exist = False
		if exist:
			# print('can be added')
			# print(str(segment))
			f_end = D[segment[1]]
			# delta_n = fi[act_index] - fj[act_index] #
			# delta_n = (segment[1] - segment[0]) / s_min
			# print('delta_n: ' + str(delta_n))
			# delta_n = int(delta_n)
			# delta_n = num[segment[1]] - num[segment[0]]
			# delta_n = 1
			# delta_array.append(delta_n)
			# delta_dictionary[str(segment)] = delta_n
			time0 = time.localtime(segment[0])
			time1 = time.localtime(segment[1])
			# print('delta_n segment' + str(ii) + ': ' + str(time0.tm_mday) + '-' + str(time1.tm_mday) + ': ' + str(delta_n))
			temp = []

			###############################
			avg = f_end
			# if delta_n == 0:
			# 	print('delta_n = 0')
			# 	for i in range(len(f_end)):
			# 		temp.append(0)
			# else:

			t_num = (segment[1] - segment[0]) / s_min
			for j in list(np.arange(segment[0], segment[1], s_min)):
				feature_v = D[j]
				for i in range(len(feature_v)):
					avg[i] += feature_v[i]

			for i in range(len(avg)):
				avg[i] = avg[i] / t_num
			#################################
			# deltaf = 0
			# for i in range(len(fi)):
				# temp.append((fi[i] - fj[i]) / delta_n)
				# temp.append((fi[i]) / delta_n)
				# temp.append((fi[i] - fj[i]))
				# temp.append(Avg / delta_n)
			for iii in range(delta_n + 1):
				point = segment[0] + (iii * s_min)
				middle_feature = D[point]
				temp.append(middle_feature)

				# deltaf += ((fi[i] - fj[i]) ** 2)
			whole_feature_matrix.append(temp)
			# print('deltaf segment' + str(ii) + ': ' + str(time0.tm_mday) + '-' + str(time1.tm_mday) + ': ' + str(deltaf ** 0.5))
		else:
			# print('can not be added')
			# print(str(segment[0]))
			# whole_feature_matrix.append(D[segment[1]])
			# fi = D[segment[1]]
			# temp = []
			# for i in range(len(fi)):
			# 	temp.append(0)
			# whole_feature_matrix.append(temp)
			print(str(segment))
			# fi = D[segment[1]]
			# fj = feature_matrix[Len]
			# fj = feature_matrix[0]
			# temp1 = smallest_segments[0]
			# print('smallest_segments[0]: ' + str(smallest_segments[0]))
			# fj = D[temp1[1]]
			# delta_n = fi[act_index] #
			delta_n = (segment[1] - segment[0]) / s_min
			# print('delta_n: ' + str(delta_n))
			delta_n = int(delta_n)
			# delta_n = 1
			# delta_n = num[segment[1]]
			delta_array.append(delta_n)
			delta_dictionary[str(segment)] = delta_n

			# Avg = fi
			# t_num = ((segment[1] - segment[0]) / s_min) - 1
			# for j in list(np.arange(segment[0] + s_min, segment[1], s_min)):
			# 	feature_v = D[j]
			# 	for i in range(len(feature_v)):
			# 		Avg[i] += feature_v[i]
			#
			# for i in range(len(Avg)):
			# 	Avg[i] = Avg[i] / t_num

			temp = []
			deltaf = 0
			# for i in range(len(fi)):
				# temp.append((fi[i] - fj[i]) / delta_n)
				# temp.append((fi[i]) / delta_n)
				# temp.append((fi[i] - fj[i]))
				# temp.append(Avg / delta_n)
			for iii in range(1, delta_n + 1):
				point = segment[0] + (iii * s_min)
				middle_feature = D[point]
				temp.append(middle_feature)

				# deltaf += ((fi[i] - fj[i]) ** 2)
			whole_feature_matrix.append(temp)
			time0 = time.localtime(segment[0])
			time1 = time.localtime(segment[1])
			# print('delta_n segment' + str(ii) + ': ' + str(time0.tm_mday) + '-' + str(time1.tm_mday) + ': ' + str(delta_n))
			# print('deltaf segment' + str(ii) + ': ' + str(time0.tm_mday) + '-' + str(time1.tm_mday) + ': ' + str(deltaf ** 0.5))
	# print('whole_feature_matrix len: ' + str(len(whole_feature_matrix)))
	# print('whole_feature_matrix len: ' + str(len(whole_feature_matrix)))
	return whole_feature_matrix, delta_dictionary, delta_array


def get_derivative(feature_matrix, all_segments, smallest_segments, num_active):
	whole_feature_matrix = []
	D = {}
	num = {}
	delta_array = []
	delta_dictionary = {}

	for i in range(len(smallest_segments)):
		small_seg = smallest_segments[i]
		D[small_seg[1]] = feature_matrix[i]
		num[small_seg[1]] = num_active[i]

	s_min = small_seg[1] - small_seg[0]
	# print('s_min: ' + str(s_min))

	ii = -1
	for segment in all_segments:
		ii += 1
		try:
			fj = D[segment[0]]
			exist = True
		except:
			exist = False
		if exist:
			# print('can be added')
			# print(str(segment))
			f_end = D[segment[1]]
			# delta_n = fi[act_index] - fj[act_index] #
			# delta_n = (segment[1] - segment[0]) / s_min
			# print('delta_n: ' + str(delta_n))
			# delta_n = int(delta_n)
			# delta_n = num[segment[1]] - num[segment[0]]
			# delta_n = 1
			# delta_array.append(delta_n)
			# delta_dictionary[str(segment)] = delta_n
			time0 = time.localtime(segment[0])
			time1 = time.localtime(segment[1])
			# print('delta_n segment' + str(ii) + ': ' + str(time0.tm_mday) + '-' + str(time1.tm_mday) + ': ' + str(delta_n))
			temp = []

			###############################
			avg = f_end
			# if delta_n == 0:
			# 	print('delta_n = 0')
			# 	for i in range(len(f_end)):
			# 		temp.append(0)
			# else:

			t_num = (segment[1] - segment[0]) / s_min
			for j in list(np.arange(segment[0], segment[1], s_min)):
				feature_v = D[j]
				for i in range(len(feature_v)):
					avg[i] += feature_v[i]

			for i in range(len(avg)):
				avg[i] = avg[i] / t_num
			#################################
			# deltaf = 0
			# for i in range(len(fi)):
				# temp.append((fi[i] - fj[i]) / delta_n)
				# temp.append((fi[i]) / delta_n)
				# temp.append((fi[i] - fj[i]))
				# temp.append(Avg / delta_n)
			for iii in range(delta_n + 1):
				point = segment[0] + (iii * s_min)
				middle_feature = D[point]
				temp.append(middle_feature)

				# deltaf += ((fi[i] - fj[i]) ** 2)
			whole_feature_matrix.append(temp)
			# print('deltaf segment' + str(ii) + ': ' + str(time0.tm_mday) + '-' + str(time1.tm_mday) + ': ' + str(deltaf ** 0.5))
		else:
			# print('can not be added')
			# print(str(segment[0]))
			# whole_feature_matrix.append(D[segment[1]])
			# fi = D[segment[1]]
			# temp = []
			# for i in range(len(fi)):
			# 	temp.append(0)
			# whole_feature_matrix.append(temp)
			# print(str(segment))
			# fi = D[segment[1]]
			# fj = feature_matrix[Len]
			# fj = feature_matrix[0]
			# temp1 = smallest_segments[0]
			# print('smallest_segments[0]: ' + str(smallest_segments[0]))
			# fj = D[temp1[1]]
			# delta_n = fi[act_index] #
			delta_n = (segment[1] - segment[0]) / s_min
			# print('delta_n: ' + str(delta_n))
			delta_n = int(delta_n)
			# delta_n = 1
			# delta_n = num[segment[1]]
			delta_array.append(delta_n)
			delta_dictionary[str(segment)] = delta_n

			# Avg = fi
			# t_num = ((segment[1] - segment[0]) / s_min) - 1
			# for j in list(np.arange(segment[0] + s_min, segment[1], s_min)):
			# 	feature_v = D[j]
			# 	for i in range(len(feature_v)):
			# 		Avg[i] += feature_v[i]
			#
			# for i in range(len(Avg)):
			# 	Avg[i] = Avg[i] / t_num

			temp = []
			deltaf = 0
			# for i in range(len(fi)):
				# temp.append((fi[i] - fj[i]) / delta_n)
				# temp.append((fi[i]) / delta_n)
				# temp.append((fi[i] - fj[i]))
				# temp.append(Avg / delta_n)
			for iii in range(1, delta_n + 1):
				point = segment[0] + (iii * s_min)
				middle_feature = D[point]
				temp.append(middle_feature)

				# deltaf += ((fi[i] - fj[i]) ** 2)
			whole_feature_matrix.append(temp)
			time0 = time.localtime(segment[0])
			time1 = time.localtime(segment[1])
			# print('delta_n segment' + str(ii) + ': ' + str(time0.tm_mday) + '-' + str(time1.tm_mday) + ': ' + str(delta_n))
			# print('deltaf segment' + str(ii) + ': ' + str(time0.tm_mday) + '-' + str(time1.tm_mday) + ': ' + str(deltaf ** 0.5))
	# print('whole_feature_matrix len: ' + str(len(whole_feature_matrix)))
	# print('whole_feature_matrix len: ' + str(len(whole_feature_matrix)))
	return whole_feature_matrix, delta_dictionary, delta_array


def get_diff(feature_matrix, all_segments, smallest_segments, num_active):
	whole_feature_matrix = []
	D = {}
	num = {}
	delta_array = []
	delta_dictionary = {}

	for i in range(len(smallest_segments)):
		small_seg = smallest_segments[i]
		D[small_seg[1]] = feature_matrix[i]
		num[small_seg[1]] = num_active[i]

	s_min = small_seg[1] - small_seg[0]
	print('s_min: ' + str(s_min))

	ii = -1
	for segment in all_segments:
		ii += 1
		try:
			fj = D[segment[0]]
			exist = True
		except:
			exist = False
		if exist:
			# print('can be added')
			print(str(segment))
			f_end = D[segment[1]]
			# delta_n = fi[act_index] - fj[act_index] #
			# delta_n = (segment[1] - segment[0]) / s_min
			# print('delta_n: ' + str(delta_n))
			# delta_n = int(delta_n)
			# delta_n = num[segment[1]] - num[segment[0]]
			# delta_n = 1
			# delta_array.append(delta_n)
			# delta_dictionary[str(segment)] = delta_n
			time0 = time.localtime(segment[0])
			time1 = time.localtime(segment[1])
			print('delta_n segment' + str(ii) + ': ' + str(time0.tm_mday) + '-' + str(time1.tm_mday) + ': ' + str(delta_n))
			temp = []

			###############################
			avg = f_end
			# if delta_n == 0:
			# 	print('delta_n = 0')
			# 	for i in range(len(f_end)):
			# 		temp.append(0)
			# else:

			t_num = (segment[1] - segment[0]) / s_min
			for j in list(np.arange(segment[0], segment[1], s_min)):
				feature_v = D[j]
				for i in range(len(feature_v)):
					avg[i] += feature_v[i]

			for i in range(len(avg)):
				avg[i] = avg[i] / t_num
			#################################
			# deltaf = 0
			# for i in range(len(fi)):
				# temp.append((fi[i] - fj[i]) / delta_n)
				# temp.append((fi[i]) / delta_n)
				# temp.append((fi[i] - fj[i]))
				# temp.append(Avg / delta_n)
			for iii in range(delta_n + 1):
				point = segment[0] + (iii * s_min)
				middle_feature = D[point]
				temp.append(middle_feature)

				# deltaf += ((fi[i] - fj[i]) ** 2)
			whole_feature_matrix.append(temp)
			print('deltaf segment' + str(ii) + ': ' + str(time0.tm_mday) + '-' + str(time1.tm_mday) + ': ' + str(deltaf ** 0.5))
		else:
			# print('can not be added')
			# print(str(segment[0]))
			# whole_feature_matrix.append(D[segment[1]])
			# fi = D[segment[1]]
			# temp = []
			# for i in range(len(fi)):
			# 	temp.append(0)
			# whole_feature_matrix.append(temp)
			print(str(segment))
			# fi = D[segment[1]]
			# fj = feature_matrix[Len]
			# fj = feature_matrix[0]
			# temp1 = smallest_segments[0]
			# print('smallest_segments[0]: ' + str(smallest_segments[0]))
			# fj = D[temp1[1]]
			# delta_n = fi[act_index] #
			delta_n = (segment[1] - segment[0]) / s_min
			print('delta_n: ' + str(delta_n))
			delta_n = int(delta_n)
			# delta_n = 1
			# delta_n = num[segment[1]]
			delta_array.append(delta_n)
			delta_dictionary[str(segment)] = delta_n

			# Avg = fi
			# t_num = ((segment[1] - segment[0]) / s_min) - 1
			# for j in list(np.arange(segment[0] + s_min, segment[1], s_min)):
			# 	feature_v = D[j]
			# 	for i in range(len(feature_v)):
			# 		Avg[i] += feature_v[i]
			#
			# for i in range(len(Avg)):
			# 	Avg[i] = Avg[i] / t_num

			temp = []
			deltaf = 0
			# for i in range(len(fi)):
				# temp.append((fi[i] - fj[i]) / delta_n)
				# temp.append((fi[i]) / delta_n)
				# temp.append((fi[i] - fj[i]))
				# temp.append(Avg / delta_n)
			for iii in range(1, delta_n + 1):
				point = segment[0] + (iii * s_min)
				middle_feature = D[point]
				temp.append(middle_feature)

				# deltaf += ((fi[i] - fj[i]) ** 2)
			whole_feature_matrix.append(temp)
			time0 = time.localtime(segment[0])
			time1 = time.localtime(segment[1])
			print('delta_n segment' + str(ii) + ': ' + str(time0.tm_mday) + '-' + str(time1.tm_mday) + ': ' + str(delta_n))
			print('deltaf segment' + str(ii) + ': ' + str(time0.tm_mday) + '-' + str(time1.tm_mday) + ': ' + str(deltaf ** 0.5))
	# print('whole_feature_matrix len: ' + str(len(whole_feature_matrix)))
	# print('whole_feature_matrix len: ' + str(len(whole_feature_matrix)))
	return whole_feature_matrix, delta_dictionary, delta_array


def get_trend(feature_matrix, all_segments, smallest_segments, num_active):
	whole_feature_matrix = []
	D = {}
	num = {}
	delta_array = []
	delta_dictionary = {}

	for i in range(len(smallest_segments)):
		small_seg = smallest_segments[i]
		D[small_seg[1]] = feature_matrix[i]
		num[small_seg[1]] = num_active[i]

	s_min = small_seg[1] - small_seg[0]
	print('s_min: ' + str(s_min))

	ii = -1
	for segment in all_segments:
		ii += 1
		delta_n = (segment[1] - segment[0]) / s_min
		delta_n = int(delta_n)
		try:
			fj = D[segment[0]]
			exist = True
		except:
			exist = False

		if exist:
			temp = []
			for iii in range(delta_n + 1):
				point = segment[0] + (iii * s_min)
				middle_feature = D[point]
				temp.append(middle_feature)

			whole_feature_matrix.append(temp)
		else:
			temp = []
			for iii in range(1, delta_n + 1):
				point = segment[0] + (iii * s_min)
				middle_feature = D[point]
				temp.append(middle_feature)

			whole_feature_matrix.append(temp)
	return whole_feature_matrix, delta_dictionary, delta_array


def get_feature(feature_matrix, all_segments, smallest_segments, num_active):
	whole_feature_matrix = []
	D = {}
	num = {}
	delta_array = []
	delta_dictionary = {}

	for i in range(len(smallest_segments)):
		small_seg = smallest_segments[i]
		D[small_seg[1]] = feature_matrix[i]
		num[small_seg[1]] = num_active[i]

	for segment in smallest_segments:
		f_end = D[segment[1]]
		whole_feature_matrix.append(f_end)
	return whole_feature_matrix, delta_dictionary, delta_array


def get_Avg(feature_matrix, all_segments, smallest_segments, num_active):
	whole_feature_matrix = []
	D = {}
	num = {}
	delta_array = []
	delta_dictionary = {}

	for i in range(len(smallest_segments)):
		acc = 1e+5
		small_seg = smallest_segments[i]
		s1 = round(small_seg[1] * acc) / acc
		D[s1] = feature_matrix[i]
		num[small_seg[1]] = num_active[i]
	s_min = small_seg[1] - small_seg[0]

	ii = -1
	for segment in all_segments:
		ii += 1
		avg = []
		temp = round(segment[1] * acc) / acc
		avg = avg + D[temp]
		t_num = ((segment[1] - segment[0]) / s_min)

		for j in list(np.arange(segment[0] + s_min, segment[1], s_min)):
			j = round(j * acc) / acc
			feature_v = D[j]
			for i in range(len(feature_v)):
				avg[i] += feature_v[i]

		for i in range(len(avg)):
			avg[i] /= t_num
		whole_feature_matrix.append(avg)
	return whole_feature_matrix, delta_dictionary, delta_array


def main(feature_matrix, all_segments, smallest_segments, num_active):
	whole_feature_matrix, delta_dictionary, delta_array = get_Avg(feature_matrix, all_segments, smallest_segments, num_active)
	return whole_feature_matrix, delta_dictionary, delta_array

if __name__ == '__main__':
	feature_matrix = sys.argv[1]
	all_segments = sys.argv[2]
	smallest_segments = sys.argv[3]
	num_active = sys.argv[4]
	mode = sys.argv[5]
	main(feature_matrix, all_segments, smallest_segments, num_active, mode)