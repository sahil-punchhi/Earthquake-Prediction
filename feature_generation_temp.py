# This is the temp file add your code here

# CURRENTLY DOING
# - add_trend_feature -> trend_adding_feature
# - classic_sta_lta   -> classic_lta_sta

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

  # ----------- End of Code ----------------
  
  
  def features(self):
    # add your section here
    pass

array_new = [12,346,5883,25,2,9,635,24,864,2,4664]
a = FeatureGenerator()
print(a.trend_adding_feature(array_new))
