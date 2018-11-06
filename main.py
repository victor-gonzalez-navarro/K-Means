import os
import re
import time

import pandas as pd
import numpy as np
from scipy.io import arff

from eval_plot.evaluation import evaluate
from eval_plot.evaluation import ploting_v
from matplotlib import pyplot
from preproc.preprocess import Preprocess
from sklearn.preprocessing.label import LabelEncoder


# ------------------------------------------------------------------------------------------------------- Read databases
def obtain_arffs(path):
    # Read all the databases
    arffs_dic = {}
    for filename in os.listdir(path):
        if re.match('(.*).arff', filename):
            arffs_dic[filename.replace('.arff', '')] = arff.loadarff(path + filename)
    return arffs_dic

# ----------------------------------------------------------------------------------------------------------------- Main
def main():
    print('\033[1m' + 'Loading all the datasets...' + '\033[0m')
    arffs_dic = obtain_arffs('./datasets/')

    # Extract an specific database
    dataset_name = 'breast-w'       # possible datasets ('hypothyroid', 'breast-w', 'waveform')
    dat1 = arffs_dic[dataset_name]
    df1 = pd.DataFrame(dat1[0])     # original data in pandas dataframe
    groundtruth_labels = df1[df1.columns[len(df1.columns)-1]].values  # original labels in a numpy array
    df1 = df1.drop(df1.columns[len(df1.columns)-1],1)
    if dataset_name == 'hypothyroid':
        df1 = df1.drop('TBG', 1)    # This column only contains NaNs so does not add any value to the clustering
    data1 = df1.values              # original data in a numpy array without labels
    load = Preprocess()
    data_x = load.preprocess_method(data1)
    le = LabelEncoder()
    le.fit(np.unique(groundtruth_labels))
    groundtruth_labels = le.transform(groundtruth_labels)

    print(len(data_x[:,0]))
    #ploting_v (data_x, 2, groundtruth_labels)
    #pyplot.scatter(data_x[:,0], data_x[:,1], s=4)
    #pyplot.show()



# ----------------------------------------------------------------------------------------------------------------- Init
if __name__ == '__main__':
    main()