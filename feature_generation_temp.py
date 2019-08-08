# This is the temp file add your code here

# CURRENTLY DOING
# - add_trend_feature -> trend_adding_feature
# - classic_sta_lta   -> lta_sta_function
# - calc_change_rate -> change_rate_calculation

# kstat_    -> k_static
# moment_   -> moments
# kstatvar_ -> variable_k_static
# features -> generate_features

# get_features()

#####################
# COMPLETED
# -

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
from collections import defaultdict

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


def generate_features(x):
    # collection of features
    feature_collection={}

    # collection of intervals
    feature_intervals={
        'k_static':list(range(0,5)),
        'variable_k_stat':[1,2]
    }

    # add your section here

    # -----------------Rishabh -----------

    for interval in feature_intervals['kstat']:
        feature_collection['k_static_{interval}']=stats.kstat(x,interval)

    for interval in feature_intervals['moment']:
        feature_collection['moments_{interval}']=stats.moment(x,interval)

    for interval in feature_intervals['variable_k_stat']:
        feature_collection['variable_k_static_{interval}']=stats.kstatvar(x,interval)

    return feature_collection

# ----------- End of Code ----------------


if __name__ == '__main__':
    array_new = np.array([12,346,5883,25,2,9,635,24,864,2,4664])
    # print(trend_adding_feature(array_new))
    # print(lta_sta_function(array_new,2,3))
    print(change_rate_calculation(array_new))
