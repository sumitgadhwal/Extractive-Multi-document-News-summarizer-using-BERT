import pandas as pd
from sklearn.cluster import KMeans
from summarizer.bert_parent import BertParent
from summarizer.sentence_handler import SentenceHandler
import os
#########

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.metrics import silhouette_score
from bs4 import BeautifulSoup






def elbow_plot(data, maxK=10, seed_centroids=None):
    """
        parameters:
        - data: pandas DataFrame (data to be fitted)
        - maxK (default = 10): integer (maximum number of clusters with which to run k-means)
        - seed_centroids (default = None ): float (initial value of centroids for k-means)
    """
    sse = {}
    for k in range(1, maxK):
        print("k: ", k)
        if seed_centroids is not None:
            seeds = seed_centroids.head(k)
            kmeans = KMeans(n_clusters=k, max_iter=500, n_init=100, random_state=0, init=np.reshape(seeds, (k,1))).fit(data)
            data["clusters"] = kmeans.labels_
        else:
            kmeans = KMeans(n_clusters=k, max_iter=300, n_init=100, random_state=0).fit(data)
            data["clusters"] = kmeans.labels_
        # Inertia: Sum of distances of samples to their closest cluster center
        sse[k] = kmeans.inertia_
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.show()
    return
    
    
    
    
    
    
    
def silhouette_plot(data, maxK=10, seed_centroids=None):
	score={}
	previous_silh_avg=0.0
	best_clusters = 0 
	for k in range(2, maxK):
		print("k: ", k)
		clusterer = KMeans(n_clusters=k, init='k-means++', random_state=0)
		cluster_labels = clusterer.fit_predict(data)
		silhouette_avg = silhouette_score(data, cluster_labels,sample_size=3000)
		score[k]=silhouette_avg
		if silhouette_avg > previous_silh_avg:
			previous_silh_avg = silhouette_avg
			best_clusters = k
	
	plt.figure()
	plt.plot(list(score.keys()), list(score.values()))
	print("No of Clusters:", best_clusters)
	plt.show()
	
	return





main_folder_path = os.getcwd() + "/Input_DUC2007/input3"
files = os.listdir(main_folder_path)

body=''
#print(files)
for file1 in files:
	f=open(main_folder_path+"/"+file1)
	raw=f.read()
	lines = BeautifulSoup(raw).find_all('p')
	line=' '
	for itr in lines:
		line=line+itr.get_text()
	#print(line)
	body=body+line
	f.close()
	#print(line)
	#print('\n\n')



sentence_handler = SentenceHandler()
sentences=sentence_handler(body)
print(len(sentences))
model = BertParent('bert-large-uncased')
hidden = model(sentences)
print(hidden.shape)

#elbow_plot(pd.DataFrame(hidden),100)
silhouette_plot(pd.DataFrame(hidden),80)


#https://github.com/analyticalmonk/KMeans_elbow
