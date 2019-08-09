# This is the temp file add your code here

# CURRENTLY DOING
# max_to_min
# max_to_min_diff
# count_big
# sum
# get_features()
# mean_change_rate
# mean_change_rate_{direction}_{slice_length}
# percentile
# abs_percentile



#####################
# COMPLETED
# -k_static
#  median_abs_dev
# variable_k_static
# kurtosis
# moments
# autocorrelation -> correlation
# skewness

# - add_trend_feature -> trend_adding_feature
# - classic_sta_lta   -> sta_lta_function
# - calc_change_rate -> change_rate_calculation
# - classic_sta_ltaN_mean -> sta_lta_mean_N
# - count_big_{slice_length}_threshold_{threshold} -> count_{slice}_greater_than_threshold_{threshold_limit}
# - trend -> linear_trend
# - abs_trend -> absolute_linear_trend
# - mean std max min
# - {agg_type}_{direction}_{slice_length} -> from_{movement_direction}_slice_{slice}_typeOfAggregation{type_of_aggregation}
# - mean_change_abs
# - abs_max
# - abs_mean
# - abs_std
# - hmean
# - gmean
from itertools import product

import numpy as np
import pandas as pd
from scipy.ndimage import mean
from sklearn.linear_model import LinearRegression
from scipy import stats
from tsfresh.feature_extraction import feature_calculators
from joblib import Parallel, delayed
import statistics

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


def generate_features(x, y, seg_id):
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

    # geometric and harminic means
    x_val = x[np.nonzero(x)[0]]
    feature_collection['geometric_mean'] = stats.gmean(np.abs(x_val))
    feature_collection['harmonic_mean'] = stats.hmean(np.abs(x_val))

    # geometric and harminic means
    feature_collection['hmean'] = stats.hmean(np.abs(x[np.nonzero(x)[0]]))
    feature_collection['gmean'] = stats.gmean(np.abs(x[np.nonzero(x)[0]]))



    # basic stats
    feature_collection['mean'] = mean(x)
    feature_collection['std'] = x.std()
    feature_collection['max'] = max(x)
    feature_collection['min'] = min(x)

    # basic stats on absolute values
    feature_collection['mean_change_abs'] = (np.diff(x)).mean()
    feature_collection['abs_max'] = max(np.abs(x))
    feature_collection['abs_mean'] = np.mean(np.abs(x))
    feature_collection['abs_std'] = np.abs(x).std()

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

    #aggregations on various slices of data
    for type_of_aggregation, movement_direction, slice in product(['std', 'mean','max', 'min'], ['last', 'first'], [50000, 10000, 1000]):
        if movement_direction == 'last':
            feature_collection[f'from_{movement_direction}_slice_{slice}_typeOfAggregation{type_of_aggregation}'] = pd.DataFrame(x[-slice:]).agg(type_of_aggregation)
        elif movement_direction == 'first':
            feature_collection[f'from_{movement_direction}_slice_{slice}_typeOfAggregation{type_of_aggregation}'] = pd.DataFrame(x[:slice]).agg(type_of_aggregation)


    # ------------------End of Code-----------------

    return feature_collection

# ----------------- Amritesh ---------------


def read_data_segments(file, dclass):
    if dclass == 'train':
        dtypes = {
            'acoustic_data': np.float64,
            'time_to_failure': np.float64
        }

        df = pd.read_csv(file, iterator=True, chunksize=150000, dtype=dtypes)

        for i, j in enumerate(df):
            x = j.acoustic_data.values
            y = j.time_to_failure.values[-1]
            seg_id = 'train_' + str(i)
            del j
            yield x, y, seg_id
    elif dclass == 'test':
        dtypes = {
            'acoustic_data': np.float64
        }

        for i, j in file:
            dfx = pd.read_csv(j, dtype=dtypes)
            x = dfx.acoustic_data.values[-150000:]
            y = -999
            del dfx
            yield x, y, i


def get_data(file, dclass):
    feature_list = []
    res = Parallel(n_jobs=4, backend='threading')(delayed(get_features)(x, y, s) for x, y, s in read_data_segments(file, dclass))

    for r in res:
        feature_list.append(r)

    return pd.DataFrame(feature_list)


def get_features(x, y, seg_id):
    zc = np.fft.fft(pd.Series(x))
    realFFT = pd.Series(np.real(zc))
    imagFFT = pd.Series(np.imag(zc))

    main_dict = generate_features(x, y, seg_id)
    r_dict = generate_features(realFFT, y, seg_id)
    i_dict = generate_features(imagFFT, y, seg_id)

    for k, v in r_dict.items():
        if k not in ['target', 'seg_id']:
            main_dict[f'fftr_{k}'] = v

    for k, v in i_dict.items():
        if k not in ['target', 'seg_id']:
            main_dict[f'ffti_{k}'] = v

    return main_dict


# -------------- END OF CODE ---------------

if __name__ == '__main__':
    file_path = './'

    train_file = file_path + 'train.csv'

    test_files = []
    submission = pd.read_csv(file_path + 'sample_submission.csv')
    for seg_id in submission.seg_id.values:
        test_files.append((seg_id, file_path + 'test/' + seg_id + '.csv'))

    get_data(train_file, 'train').to_csv(file_path + 'train_df.csv', index=False)
    get_data(test_files, 'test').to_csv(file_path + 'test_df.csv', index=False)
