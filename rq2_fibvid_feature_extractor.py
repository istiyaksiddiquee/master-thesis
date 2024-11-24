import pickle
import warnings
import pandas as pd 
from tqdm import tqdm
from time import time

from feature_extractor import FeatureExtractor

warnings.simplefilter(action='ignore')

start_time = time()

result = pd.DataFrame({
                'cascade_root_id': [],
                'category': [],
                'depth' : [], 
                'size' : [], 
                'max_breadth' : [], 
                'virality' : [], 
                'strongly_cc' : [], 
                'weakly_cc' : [], 
                'size_of_scc' : [], 
                'avg_cluster_coef' : [], 
                'density' : [], 
                'layer_ratio' : [], 
                'structural_heterogeneity' : [], 
                'characteristic_distance' : [], 
                'veracity': [],
            })

c = 0
map = {}
truth_value = [True, False, True, False]
categories = ['covid_true', 'covid_false', 'non_covid_true', 'non_covid_false']

for category in categories:

    pickle_file_path = './fibvid_' + category +'_root_to_graph.pickle'

    with open(pickle_file_path, 'rb') as handle:
        map = pickle.load(handle)

    print('loaded map with length: ', len(map))

    for map_entry in tqdm(map.keys()):

        cascade_root_id, veracity = map_entry.split('_')
        graph = map.get(map_entry)
        
        feature_extrator = FeatureExtractor(graph = graph)
        entry = (cascade_root_id, category, ) + feature_extrator.extract_features(root=int(cascade_root_id)) + (truth_value[c], )
        result.loc[len(result.index)] = list(entry)
    
    c += 1

result.to_csv('./rq2_fibvid_features.csv', sep=',', encoding='utf-8')
end_time = time()
print("Took : {} minutes".format((end_time-start_time)/60))