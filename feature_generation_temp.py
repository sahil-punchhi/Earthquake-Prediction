# This is the temp file add your code here

# CURRENTLY DOING
# - add_trend_feature -> trend_adding_feature
# - classic_sta_lta   -> sta_lta_function
# - calc_change_rate -> change_rate_calculation
# - classic_sta_ltaN_mean -> sta_lta_mean_N
# - count_big_{slice_length}_threshold_{threshold} -> count_{slice}_greater_than_threshold_{threshold_limit}
# - trend -> linear_trend
# - abs_trend -> absolute_linear_trend
# - {agg_type}_{direction}_{slice_length} ->

# get_features()

#####################
# COMPLETED
# -k_static
#  median_abs_dev
# variable_k_static
# kurtosis
# moments
# autocorrelation -> correlation
# skewness

from itertools import product

import os
import numpy as np
import pandas as pd
from scipy.ndimage import mean
from sklearn.linear_model import LinearRegression
from scipy import stats
from tsfresh.feature_extraction import feature_calculators
from joblib import Parallel, delayed
import statistics
import multiprocessing as mp

# ----------------- Aarushi -----------


def trend_adding_feature(array, absolute=False):
    arr_len = len(array)
    index = np.array(range(arr_len))

    if absolute:
        array = np.abs(array)

    lr = LinearRegression().fit(index.reshape(-1, 1), array)

    return lr.coef_[0]


def sta_lta_function(x, length_sta, length_lta):
    x_sq = x ** 2
    sta = np.cumsum(x_sq)

    # Convert to float
    lta = np.require(sta, dtype=np.float)

    # Copy for LTA
    sta = lta.copy()

    # Compute the STA and the LTA
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta /= length_lta

    # Pad zeros
    for i in range(0, length_lta-1):
        sta[i] = 0

    # Avoid division by zero by setting zero values to tiny float
    tiny_val = np.finfo(0.0).tiny
    index_new = lta < tiny_val
    lta[index_new] = tiny_val
    return_val = sta / lta

    return return_val


def change_rate_calculation(x):
    x_ = np.diff(x)
    change_val = (x_ / x[:-1])
    change_val = change_val[(change_val != 0)]
    change_val = change_val[np.isfinite(change_val)]
    return_val = np.mean(change_val)

    return return_val

# ----------- End of Code ----------------


def generate_features(x):
    # collection of features
    feature_collection = {}

    # collection of intervals
    feature_intervals={
        'k_static':list(range(1,5)),
        'variable_k_static':[1,2],
        'auto_lags':[5, 10, 50, 100, 500, 1000, 5000, 10000]
    }

    # add your section here

    # -----------------Rishabh -----------

    for interval in feature_intervals['k_static']:
        feature_collection['k_static_{interval}'] = stats.kstat(x, interval)

    feature_collection['median_abs_dev'] = stats.median_absolute_deviation(x)

    for interval in feature_intervals['variable_k_static']:
        feature_collection['variable_k_static_{interval}'] = stats.kstatvar(x, interval)

    feature_collection['kurtosis'] = stats.kurtosis(x)

    for interval in feature_intervals['k_static']:
        feature_collection['moments_{interval}'] = stats.moment(x, interval)

    feature_collection['median'] = statistics.median(x)

    feature_collection['skewness'] = stats.skew(x)

    for interval in feature_intervals['auto_lags']:
        feature_collection['correlation_{interval}']=feature_calculators.autocorrelation(x, interval)

    # ----------- End of Code ----------------

    # -----------------Aarushi-------------------
    feature_collection['sta_lta_mean_1'] = mean(sta_lta_function(x, 500, 10000))
    feature_collection['sta_lta_mean_2'] = mean(sta_lta_function(x, 4000, 10000))
    feature_collection['sta_lta_mean_3'] = mean(sta_lta_function(x, 5000, 100000))
    feature_collection['sta_lta_mean_4'] = mean(sta_lta_function(x, 333, 666))
    feature_collection['sta_lta_mean_5'] = mean(sta_lta_function(x, 3333, 6666))
    feature_collection['sta_lta_mean_6'] = mean(sta_lta_function(x, 100, 5000))
    feature_collection['sta_lta_mean_7'] = mean(sta_lta_function(x, 50, 1000))
    feature_collection['sta_lta_mean_8'] = mean(sta_lta_function(x, 10000, 25000))

    feature_collection['linear_trend'] = trend_adding_feature(x)
    feature_collection['absolute_linear_trend'] = trend_adding_feature(x, absolute =True)

    for slice, threshold_limit in product([50000, 100000, 150000], [5, 10, 20, 50, 100]):
        x_sliced = np.abs(x[-slice:])
        feature_collection[f'count_{slice}_greater_than_threshold_{threshold_limit}'] = (x_sliced > threshold_limit).sum()
        feature_collection[f'count_{slice}_less_than_threshold_{threshold_limit}'] = ( x_sliced < threshold_limit).sum()
    # ------------------End of Code-----------------

    return feature_collection


# ----------------- Amritesh ---------------


def create_feature_dict(x, y=None):
    dft = np.fft.fft(pd.Series(x))
    feature_collection = generate_features(x)

    for key, val in generate_features(pd.Series(np.real(dft))).items():
        feature_collection[f'fft_real_{key}'] = val

    for key, val in generate_features(pd.Series(np.imag(dft))).items():
        feature_collection[f'fft_imag_{key}'] = val

    if y:
        return feature_collection, y

    return feature_collection


def get_train_segments(file):
    dtypes = {
        'acoustic_data': np.int16,
        'time_to_failure': np.float32
    }

    for i in pd.read_csv(file, dtype=dtypes, iterator=True, chunksize=150000):
        yield i['acoustic_data'].to_numpy(), i['time_to_failure'].to_numpy()[-1]


def get_test_segments(file):
    dtypes = {
        'acoustic_data': np.int16
    }

    for i in file:
        df = pd.read_csv(i, dtype=dtypes)
        yield df['acoustic_data'].to_numpy()


def get_train_data(file, cores):
    features = Parallel(n_jobs=cores, backend='threading')(delayed(create_feature_dict)(x, y) for x, y in get_train_segments(file))

    return pd.DataFrame([f[0] for f in features]), pd.DataFrame([f[1] for f in features])


def get_test_data(file, cores):
    features = Parallel(n_jobs=cores, backend='threading')(delayed(create_feature_dict)(x) for x in get_test_segments(file))

    return pd.DataFrame([f for f in features])


def preprocessing(path):
    num_cores = mp.cpu_count()

    train_file = path + 'train.csv'
    xtrain, ytrain = get_train_data(train_file, num_cores)
    xtrain.to_csv(path + 'train_df.csv', index=False)

    test_files = [path + 'test/' + i + '.csv' for i in (pd.read_csv(file_path + 'sample_submission.csv'))['seg_id'].tolist()]
    xtest = get_test_data(test_files, num_cores)
    xtest.to_csv(path + 'test_df.csv', index=False)

    return xtrain, ytrain, xtest

# -------------- END OF CODE ---------------


if __name__ == '__main__':
    file_path = os.getcwd() + '/data_files/'

    X_train, y_train, X_test = preprocessing(file_path)
