import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn import metrics as mt

# To evaluate the clustering algorithms
def evaluate(labels_method, groundtruth_labels, data_x = None):
    n_instances = len(groundtruth_labels)
    f00 = 0; f01 = 0; f10 = 0; f11 = 0;

    for i in range(0, n_instances):
        for j in range(i + 1, n_instances):
            # Different class, different cluster
            if (groundtruth_labels[i] != groundtruth_labels[j]) and (labels_method[i] != labels_method[j]):
                f00 += 1
            # Different class, same cluster
            elif (groundtruth_labels[i] != groundtruth_labels[j]) and (labels_method[i] == labels_method[j]):
                f01 += + 1
            # Same class, different cluster
            elif (groundtruth_labels[i] == groundtruth_labels[j]) and (labels_method[i] != labels_method[j]):
                f10 += + 1
            # Same class, same cluster
            elif (groundtruth_labels[i] == groundtruth_labels[j]) and (labels_method[i] == labels_method[j]):
                f11 += + 1

    score_randstatistic = (f00 + f11) / (f00 + f01 + f10 + f11)
    print('The Rand Statistic score is: ' + '\033[1m'+'\033[94m'+str(round(score_randstatistic,3))+'\033[0m')

    score_calinski = mt.calinski_harabaz_score(data_x, labels_method.reshape(len(labels_method), ))
    print('The Calinski Harabaz score is: ' + '\033[1m'+'\033[94m'+str(round(score_calinski,3))+'\033[0m')

    #score_jaccardcoefficient = f11 / (f01 + f10 + f11)
    #print('The Jaccard Coefficient score is: ' + '\033[1m'+'\033[94m'+str(round(score_jaccardcoefficient,3))+'\033[0m')


# To do a scatter plot of the final result of the kmeans algorithm
def ploting_v(data_x, n_clusters, lista):
    n_features = data_x.shape[1]
    wh = []
    for i in range(n_clusters):
        wh.append(np.argwhere(np.array(lista) == i))

    fig = plt.figure(figsize=(n_features + 3, n_features + 3))
    suma = 0
    for k1 in range(n_features):
        for k2 in range(k1, n_features):
            suma = n_features * k1 + k2 + 1
            color = iter(cm.rainbow(np.linspace(0, 1, n_clusters)))
            ax = fig.add_subplot(n_features, n_features, suma)
            for i in range(n_clusters):
                c = next(color)
                #ax.plot(data_x[wh[i], k1], data_x[wh[i], k2], ',', 1, c= c)
                ax.scatter(data_x[wh[i], k1], data_x[wh[i], k2], s=4)

    plt.show()