# This is the temp file add your code here

# CURRENTLY DOING
# - add_trend_feature -> trend_adding_feature
# - classic_sta_lta   -> lta_sta_function
# - calc_change_rate -> change_rate_calculation

# kstat_    -> k_static
# moment_   -> moments
# kstatvar_ -> variable_k_static

# get_features()

#####################
# COMPLETED
# -

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats
from joblib import Parallel, delayed


# ----------------- Aarushi -----------

def trend_adding_feature(array, absolute=False):
    arr_len = len(array)
    index = np.array(range(arr_len))

    if absolute:
        array = np.abs(array)

    lr = LinearRegression().fit(index.reshape(-1, 1), array)

    return lr.coef_[0]


def lta_sta_function(x, length_sta, length_lta):
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
    feature_intervals = {
        'k_static': list(range(1, 5)),
        'variable_k_stat': [1, 2]
    }

    # add your section here

    # -----------------Rishabh -----------

    for interval in feature_intervals['k_static']:
        feature_collection['k_static_{interval}'] = stats.kstat(x, interval)


    feature_collection['mean_abs_dev'] = stats.median_absolute_deviation(x)


    for interval in feature_intervals['variable_k_stat']:
        feature_collection['variable_k_static_{interval}'] =     stats.kstatvar(x, interval)

    # feature_collection['kurtosis'] = x.kurtosis()

    for interval in feature_intervals['k_static']:
        feature_collection['moments_{interval}'] = stats.moment(x, interval)

    # feature_collection['median'] = x.median()

    # feature_collection['skewness'] = x.skew()

    # ----------- End of Code ----------------

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
    file_path = 'C:\\Users\\amrit\\OneDrive\\Desktop\\files\\'

    train_file = file_path + 'train.csv'

    test_files = []
    submission = pd.read_csv(file_path + 'sample_submission.csv')
    for seg_id in submission.seg_id.values:
        test_files.append((seg_id, file_path + 'test\\' + seg_id + '.csv'))

    get_data(train_file, 'train').to_csv(file_path + 'train_df.csv', index=False)
    get_data(test_files, 'test').to_csv(file_path + 'test_df.csv', index=False)
