# This is the temp file add your code here

# CURRENTLY DOING
# - add_trend_feature -> trend_adding_feature
# - classic_sta_lta   -> lta_sta_function
# - calc_change_rate -> change_rate_calculation

# kstat_    -> kstat
# moment_   -> moment
# kstatvar_ -> variable_k_stat

# basic stats
# basic stats on absolute values
# geometric and harmonic means

#####################
# COMPLETED
# -

import numpy as np
from sklearn.linear_model import LinearRegression
from collections import defaultdict


# -----------------Aarushi -----------


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


def features():



def change_rate_calculation(x):
    x_ = np.diff(x)
    change_val = (x_ / x[:-1])
    change_val = change_val[(change_val != 0)]
    change_val = change_val[np.isfinite(change_val)]
    return_val = np.mean(change_val)

    return return_val

# ----------- End of Code ----------------


class FeatureGenerator(object):
    def __init__(self):
        pass

    def features(self):



if __name__ == '__main__':
    array_new = np.array([12, 346, 5883, 25, 2, 9, 635, 24, 864, 2, 4664])
    # print(trend_adding_feature(array_new))
    # print(lta_sta_function(array_new,2,3))
    print(change_rate_calculation(array_new))
