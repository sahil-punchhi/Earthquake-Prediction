# This is the temp file add your code here

# CURRENTLY DOING
# - add_trend_feature -> trend_adding_feature
# - classic_sta_lta   -> lta_sta_function

 #####################
# COMPLETED
# -

import numpy as np
from sklearn.linear_model import LinearRegression

class FeatureGenerator(object):
  def __init__(self):
    pass

  # -----------------Aarushi -----------

  def trend_adding_feature(self, array, absolute=False):
    arr_len = len(array)
    index = np.array(range(arr_len))
    if absolute:
      array = np.abs(array)
    lr = LinearRegression().fit(index.reshape(-1, 1), array)
    return lr.coef_[0]


  def lta_sta_function(self, x, length_sta, length_lta):
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

  # ----------- End of Code ----------------
  
  
  def features(self):
    # add your section here
    pass

array_new = np.array([12,346,5883,25,2,9,635,24,864,2,4664])
a = FeatureGenerator()
#print(a.trend_adding_feature(array_new))
print(a.lta_sta_function(array_new,2,3))
