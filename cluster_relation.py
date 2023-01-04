import numpy as np
from scipy.io import loadmat 
from sklearn.cluster import KMeans
import json
from glob import glob
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Clustering the pairwise relationship between joints')
parser.add_argument('--data', type=str, default=None, help='The Mat file storing joint information')
parser.add_argument('--human_exp_dir', type=str, default=None, help='Directory storing data for human experiments')
parser.add_argument('--num_cluster', type=int, default=11, help='Number of pairwise relationship between two joints')

args = parser.parse_args()

# only 6 types of joints are considered
id2joint = {0:'ankle',
			1: 'knee',
			2: 'hip',
			3: 'hip',
			4: 'knee',
			5: 'ankle',
			6: 'wrist',
			7: 'elbow',
			8: 'shoulder',
			9: 'shoulder',
			10: 'elbow',
			11: 'wrist'
			}

joint2id = {'ankle': [0, 5],
			'knee': [1, 4],
			'hip': [2, 3],
			'wrist': [6, 11],
			'elbow': [7, 10],
			'shoulder': [8,9]
}

joint_connectivity= {
					'ankle': ['knee'],
					'knee': ['ankle', 'hip'],
					'hip': ['knee', 'shoulder'],
					'wrist': ['elbow'],
					'elbow': ['shoulder', 'wrist'],
					'shoulder': ['hip', 'elbow']
					}

id_connectivity = {0: [1],
				1: [0, 2],
				2: [1, 8],
				3: [4, 9],
				4: [3, 5],
				5: [4],
				6: [7],
				7: [6, 8],
				8: [2, 7],
				9: [3, 10],
				10: [9, 11],
				11: [10]	
				}

def main():
	# loading mat file in Python is a bit tricky
	annotation = loadmat(args.data)[os.path.basename(args.data)[:-4]]

	# recording images reserved for human experiments
	human_exp = dict()
	exp_pool = glob(os.path.join(args.human_exp_dir, '*'))
	for exp in exp_pool:
		data_name = glob(os.path.join(exp, '*.csv'))
		data = pd.read_csv(data_name[0])
		for img_id in data['img_name']:
			img_id = int(img_id.split('_')[0])
			human_exp[img_id] = 1

	# the relationships are directional, thus iterate through different source and target
	cluster_center = dict()
	processed_data = dict()
	score = [] 
	for source in joint2id:
		for target in joint2id:
			if target == source or target not in joint_connectivity[source]:
				continue

			# the experiments do not differentiate left/right joints
			training_data = []				
			for source_id in joint2id[source]:
				for target_id in joint2id[target]:
					# only consider pairwise relationship between connected joints
					if target_id not in id_connectivity[source_id]:
						continue

					# iterate through training data (exclude those for human experiments)
					for idx in range(2000):
						if (idx+1) in human_exp:
							continue
						source_x = annotation[0, source_id, idx] # originally [x, y]
						source_y = annotation[1, source_id, idx]
						target_x = annotation[0, target_id, idx]
						target_y = annotation[1, target_id, idx]
						delta_x = source_x-target_x
						delta_y = source_y-target_y

						# clustering based on relative position
						training_data.append([delta_x, delta_y])

						# clustering based on relative angle
						# angle = np.arctan(delta_y/(delta_x+1e-5))*180/np.pi
						# training_data.append([angle])

						# # clustering based on both angle and position
						# angle = np.arctan(delta_y/(delta_x+1e-5))*180/np.pi
						# training_data.append([angle, delta_x, delta_y])

			print(source, target, len(training_data))
			# apply KMeans clustering
			model = KMeans(n_clusters=args.num_cluster).fit(training_data)
			score.append(model.inertia_)
			# record the cluster center
			cluster_center[source+'_'+target] = model.cluster_centers_

			# assign label to all data (including human data)
			for idx in range(2000):
				if idx+1 not in processed_data:
					processed_data[idx+1] = dict()
				for source_id in joint2id[source]:
					if source_id not in processed_data[idx+1]:
						processed_data[idx+1][source_id] = dict()

					for target_id in joint2id[target]:
						if target_id not in id_connectivity[source_id]:
							continue
						source_x = annotation[0, source_id, idx]
						source_y = annotation[1, source_id, idx]
						target_x = annotation[0, target_id, idx]
						target_y = annotation[1, target_id, idx]
						delta_x = source_x-target_x
						delta_y = source_y-target_y

						# for relative position
						pred_label = model.predict([[delta_x, delta_y]])

						# # for relative angle
						# angle = np.arctan(delta_y/(delta_x+1e-5))*180/np.pi
						# pred_label = model.predict([[angle]])

						# # for both angle and position
						# angle = np.arctan(delta_y/(delta_x+1e-5))*180/np.pi
						# pred_label = model.predict([[angle, delta_x, delta_y]])

						processed_data[idx+1][source_id][target_id] = int(pred_label[0])

	print('Clustering score: %.3f' %np.mean(score))

	# sort the cluster assignment for better interpretability
	for idx in processed_data:
		for source_id in processed_data[idx]:
			processed_data[idx][source_id] = {k:v for k,v in sorted(
							processed_data[idx][source_id].items(), key=lambda item: item[0])}

	# instead of formulating the relationship for multi-label classification, use it for simple categorization
	rel2cat = dict()
	total_cat = 0
	for joint in joint2id:
		rel2cat[joint] = dict()
		for source_id in joint2id[joint]:
			for idx in processed_data:
				cur_rel = [source_id] + list(processed_data[idx][source_id].values())
				cur_rel = tuple(cur_rel)
				if cur_rel not in rel2cat[joint]:
					rel2cat[joint][cur_rel] = len(rel2cat[joint])
		total_cat += len(rel2cat[joint])

	print('Totally %d relationship after categorization' %total_cat)

	with open('./data/cluster_assignment.json', 'w') as f:
		json.dump(processed_data, f)
	np.save('./data/cluster_center', cluster_center)
	np.save('./data/cluster_assignment_categorized', rel2cat)
	with open('./data/excluded_human_data.json', 'w') as f:
		json.dump(human_exp, f)

main()

