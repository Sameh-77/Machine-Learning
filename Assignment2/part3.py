#======================================================#
#== Sameh Algharabli - CNG 409 -- Assignment2 -- part3 ==#
#======================================================#
#--- Libraries ---#

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
#---------------------------------------------------#

def plot_dendogram(model, **kwargs):
    #create linkage matrix and then plot the dendogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1 # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    distances = counts
    linkage_matrix = np.column_stack([model.children_, distances,  counts]).astype(float)

    #plot the corresponding dendogram
    dendrogram(linkage_matrix, **kwargs)
#------------------------------------------------------------

dataset = pickle.load(open("../data/part3_dataset.data", "rb"))
distance_metric = ["cosine","euclidean"]
linkage_creterion = ["single","complete"]
k_values = [2,3,4,5]

results = []
for k in k_values:
    for metric in distance_metric:
        for linkage in linkage_creterion:
            print("Running for k = {}, metric = {}, linkage = {}".format(k,metric,linkage))
            hac = AgglomerativeClustering(distance_threshold=None, n_clusters=k, affinity=metric,linkage=linkage )
            #hac.fit(dataset)

            #visualizer = SilhouetteVisualizer(hac, colors="yellowbrick",is_fitted=True)

            #visualizer.fit(dataset)
            hac.fit(dataset)
            #silhouette_average = visualizer.silhouette_score_
            labels = hac.labels_
            silhouette_average = silhouette_score(dataset,labels)

            hac_info = {"k": k, "metric": metric, "linkage": linkage, "silhouette_average": silhouette_average}
            results.append(hac_info)
            print("silhouette_average = ", silhouette_average)
            print("Completed...")

            print("============================================")
            plot_dendogram(hac, truncate_mode = "level", p =10)
            plt.xlabel('data')
            plt.ylabel('clusters')
            title = "Dendrogram for k = " + str(k) + ", metric = " + metric + ", linkage = " + linkage
            plt.title(title)
            plt.show()

# Finding the best highest silhouette_average and the best parameters #
best_k = 0
best_metric = 0
best_linkage = 0
max_silhouette = 0
for hacc in results:
    if hacc['silhouette_average']>max_silhouette:
        max_silhouette = hacc['silhouette_average']
        best_metric = hacc['metric']
        best_linkage = hacc['linkage']
        best_k = hacc['k']
print("#--------------------------------------#")
print("The best parameters that scored the highest silhouette_average are: ")
print("Best silhouette_average: ", max_silhouette)
print("Best_k: ", best_k)
print("Best_metric: ", best_metric)
print("Best_Linkage: ", best_linkage)
#---------------------------