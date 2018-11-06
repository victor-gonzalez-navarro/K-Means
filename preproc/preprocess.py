import numpy as np
import pandas as pd


class Preprocess:

    def preprocess_method(self, data):
        features_del = []

        for feature in range(data.shape[1]):

            # Numerical Features
            if type(data[0, feature]) in [float, np.float64]:

                # Calculate the mean of this feature of feed NaNs with it
                mean_v = np.nanmean(data[:, feature], dtype=float)

                # Calculate the max and min to normalize numerical data between 0 and 1
                max_v = np.nanmax(data[:, feature])
                min_v = np.nanmin(data[:, feature])

                for sample in range(data.shape[0]):
                    if np.isnan(data[sample, feature]):
                        data[sample, feature] = mean_v
                    if max_v != 0:
                        data[sample, feature] = (data[sample, feature] - min_v) / (max_v - min_v)

            # Categorical Features
            if type(data[0, feature]) is bytes:

                # Calculate the mode of this feature
                cat_values = np.unique(data[:, feature])
                moda = max(cat_values, key=lambda x: data[:,feature].tolist().count(x))

                # Assign the mode to NaNs
                cond_nan = np.where(data[:, feature] == '?'.encode('utf8'))
                data[cond_nan, feature] = moda

                # OneHotEncoding
                data1 = np.array(pd.get_dummies(data[:,feature]))
                data = np.concatenate((data, data1), axis=1)

                features_del.append(feature)

        # Delete categorical feature
        data = np.delete(data, features_del, 1)

        return data
