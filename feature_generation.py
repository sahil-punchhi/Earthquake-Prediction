import numpy as np
import pandas as pd
import statistics
import warnings
from datetime import datetime
from scipy.ndimage import mean
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats
from tsfresh.feature_extraction import feature_calculators
from joblib import Parallel, delayed
from itertools import product

warnings.simplefilter(action='ignore', category=FutureWarning)


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
    lta = np.require(sta, dtype=np.float64)

    # Copy for LTA
    sta = lta.copy()

    # Compute the STA and the LTA
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta /= length_lta

    # Pad zeros
    for i in range(0, length_lta - 1):
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


def generate_features(x):
    # collection of features
    feature_collection = {}

    # collection of intervals
    feature_intervals = {
        'k_static': list(range(1, 5)),
        'variable_k_static': [1, 2]
    }

    for interval in [50, 10, 100, 20]:
        feature_collection[f'discrimination_power_{interval}'] = feature_calculators.c3(x, interval)

    for interval in [500, 10000, 1000, 10, 50, 100]:
        standard_dev = pd.DataFrame(x).rolling(interval).std().dropna().values

        for sub_interval in [50, 60, 70, 75, 1, 40, 80, 90, 95, 99, 5, 10, 20, 25, 30]:
            feature_collection[f'{interval}_{sub_interval}_standard_percentile'] = np.percentile(standard_dev, sub_interval)

    for interval in feature_intervals['k_static']:
        feature_collection[f'{interval}_k_static'] = stats.kstat(x, interval)

    feature_collection['median_abs_dev'] = stats.median_absolute_deviation(x)

    for interval in feature_intervals['variable_k_static']:
        feature_collection[f'{interval}_variable_k_static'] = stats.kstatvar(x, interval)

    feature_collection['kurtosis'] = stats.kurtosis(x)

    for interval in feature_intervals['k_static']:
        feature_collection[f'{interval}_moments'] = stats.moment(x, interval)

    feature_collection['median'] = statistics.median(x)

    feature_collection['skewness'] = stats.skew(x)

    for interval in [1000, 5000, 10000, 5, 10, 50, 100, 500]:
        feature_collection[f'{interval}_correlation'] = feature_calculators.autocorrelation(x, interval)

    for interval in [50, 10, 100, 20]:
        feature_collection[f'{interval}_peak_number'] = feature_calculators.number_peaks(x, interval)

    # geometric and harmonic means
    x_val = x[x.to_numpy().nonzero()[0]]
    feature_collection['geometric_mean'] = stats.gmean(np.abs(x_val))
    feature_collection['harmonic_mean'] = stats.hmean(np.abs(x_val))

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

    percentile_divisions = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]

    for p in percentile_divisions:
        feature_collection[f'{p}th_abs_percentile'] = np.percentile(np.abs(x), p)
        feature_collection[f'{p}th_percentile'] = np.percentile(x, p)

    feature_collection['maximum_absoluteMinimum_ratio'] = max(x) / np.abs(min(x))
    feature_collection['diff_maximum_and_minimum'] = max(x) - np.abs(min(x))
    feature_collection['x_sum'] = x.sum()
    feature_collection['count_x_greater_than_500_BIG'] = len(x[np.abs(x) > 500])

    feature_collection['max_to_min'] = x.max() / np.abs(x.min())
    feature_collection['max_to_min_diff'] = x.max() - np.abs(x.min())
    feature_collection['count_big'] = len(x[np.abs(x) > 500])
    feature_collection['sum'] = x.sum()

    feature_collection['valid_mean_change_rate'] = change_rate_calculation(x)

    # calc_change_rate on slices of data
    for slice, movement_direction in product([50000, 1000, 1000], ['last', 'first']):
        if movement_direction == 'last':
            x_sliced = x[-slice:]
            feature_collection[f'from_{movement_direction}_slice_{slice}_valid_mean_change_rate'] = change_rate_calculation(x_sliced)
        elif movement_direction == 'first':
            x_sliced = x[:slice]
            feature_collection[f'from_{movement_direction}_slice_{slice}_valid_mean_change_rate'] = change_rate_calculation(x_sliced)

    for slice_length, direction in product([50000, 1000, 1000], ['last', 'first']):
        if direction == 'first':
            feature_collection[f'mean_change_rate_{direction}_{slice_length}'] = change_rate_calculation(x[:slice_length])
        elif direction == 'last':
            feature_collection[f'mean_change_rate_{direction}_{slice_length}'] = change_rate_calculation(x[-slice_length:])

    feature_collection['linear_trend'] = trend_adding_feature(x)
    feature_collection['absolute_linear_trend'] = trend_adding_feature(x, absolute=True)

    for slice, threshold_limit in product([50000, 100000, 150000], [5, 10, 20, 50, 100]):
        x_sliced = np.abs(x[-slice:])
        feature_collection[f'count_{slice}_greater_than_threshold_{threshold_limit}'] = (x_sliced > threshold_limit).sum()
        feature_collection[f'count_{slice}_less_than_threshold_{threshold_limit}'] = (x_sliced < threshold_limit).sum()

    # aggregations on various slices of data
    for type_of_aggregation, movement_direction, slice in product(['std', 'mean', 'max', 'min'], ['last', 'first'], [50000, 10000, 1000]):
        if movement_direction == 'last':
            feature_collection[f'from_{movement_direction}_slice_{slice}_typeOfAggregation{type_of_aggregation}'] = pd.DataFrame(x[-slice:]).agg(type_of_aggregation)[0]
        elif movement_direction == 'first':
            feature_collection[f'from_{movement_direction}_slice_{slice}_typeOfAggregation{type_of_aggregation}'] = pd.DataFrame(x[:slice]).agg(type_of_aggregation)[0]

    return feature_collection


def create_feature_dict(x, y=None):
    dft = np.fft.fft(pd.Series(x))
    feature_collection = generate_features(pd.Series(x))

    for key, val in generate_features(pd.Series(np.real(dft))).items():
        feature_collection[f'fft_real_{key}'] = val

    for key, val in generate_features(pd.Series(np.imag(dft))).items():
        feature_collection[f'fft_imag_{key}'] = val

    if y:
        return feature_collection, y

    return feature_collection


def get_train_segments(file):
    dtypes = {
        'acoustic_data': np.float64,
        'time_to_failure': np.float64
    }

    for i in pd.read_csv(file, dtype=dtypes, iterator=True, chunksize=150000):
        yield i['acoustic_data'].to_numpy(), i['time_to_failure'].to_numpy()[-1]


def get_test_segments(file):
    dtypes = {
        'acoustic_data': np.float64
    }

    for i in file:
        df = pd.read_csv(i, dtype=dtypes)
        yield df['acoustic_data'].to_numpy()


def get_train_data(file, cores):
    features = Parallel(n_jobs=cores, backend='threading')(delayed(create_feature_dict)(x, y) for x, y in get_train_segments(file))

    return pd.DataFrame([f[0] for f in features], dtype=np.float64), pd.Series([f[1] for f in features], dtype=np.float64)


def get_test_data(file, cores):
    features = Parallel(n_jobs=cores, backend='threading')(delayed(create_feature_dict)(x) for x in get_test_segments(file))

    return pd.DataFrame([f for f in features], dtype=np.float64)


def scale_data(data):
    sc = StandardScaler()

    sc.fit(data[0])
    scaled_xtrain = pd.DataFrame(sc.transform(data[0]), columns=data[0].columns)

    sc.fit(data[1])
    scaled_xtest = pd.DataFrame(sc.transform(data[1]), columns=data[1].columns)

    return scaled_xtrain, scaled_xtest


def fill_missing_vals(data):
    means_dict = dict()

    for i in data[0].columns:
        if data[0][i].isnull().any():
            means_dict[i] = data[0].loc[data[0][i] != -np.inf, i].mean()
            data[0][i] = data[0][i].fillna(means_dict[i])
            data[0].loc[data[0][i] == -np.inf, i] = means_dict[i]

    for i in data[1].columns:
        if data[1][i].isnull().any():
            data[1][i] = data[1][i].fillna(means_dict[i])
            data[1].loc[data[1][i] == -np.inf, i] = means_dict[i]

    return data[0], data[1]


def preprocessing(path):
    ti = datetime.now()
    num_cores = 20

    train_file = path + 'train.csv'
    xtrain, ytrain = get_train_data(train_file, num_cores)

    test_files = [path + 'test/' + i + '.csv' for i in (pd.read_csv(path + 'sample_submission.csv'))['seg_id'].tolist()]
    xtest = get_test_data(test_files, num_cores)

    xtrain, xtest = scale_data(fill_missing_vals((xtrain, xtest)))

    return xtrain, ytrain, xtest, ti
