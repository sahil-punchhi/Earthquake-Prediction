# http://hyperphysics.phy-astr.gsu.edu/hbase/Math/fft.html

import gc
import os
import time
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

path = "/Users/sahilpunchhi/Desktop/COMP 9417 ML/Project"
os.chdir(path)
train_df = pd.read_csv(os.path.join(path, 'train.csv'), dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})


# divide train data according to test data and create segments
# rows = 150000
# segments = int(np.floor(train_df.shape[0] / rows))
# print("Number of segments: ", segments)

##################################################################
# DATA EXPLORATION
##################################################################

# function to plot the single data field
# first argument: data field1, second argument: title of plot, third argument: data label
# univariate distribution plot (distplot() function in seaborn library)
# this draws a histogram and fits a kernel density estimate (KDE)
def plot_data_fields(data_field, title, data_label):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    axis = sns.distplot(data_field, kde=True, fit=stats.norm)
    axis.set_xlabel(data_label)
    plt.grid(True)
    plt.show()


# function to plot the interaction of data fields
# first argument: data field1 on left y axis, second argument: data field2 on right y axis, third argument: type of sampling used for the data
def visualize_experimental_data(acoustic_data, time_to_failure_data, sample_type):
    figure, axis1 = plt.subplots(figsize=(10, 6))
    plt.title("Seismic Signal and Time to Failure : " + sample_type)
    plt.plot(acoustic_data, color='orange')
    axis1.set_ylabel('seismic signal')
    plt.legend(['seismic signal'], loc=(0.01, 0.95))

    axis2 = axis1.twinx()
    plt.plot(time_to_failure_data, color='blue')
    axis2.set_ylabel('time to failure')
    plt.legend(['time to failure'], loc=(0.01, 0.9))
    plt.grid(True)
    plt.show()


# no of data points in train data
number_of_rows_train = train_df.shape[0]
number_of_columns_train = train_df.shape[1]

# description of training data
train_df.acoustic_data.describe()
train_df.time_to_failure.describe()

# display top 10 rows of training data
pd.options.display.precision = 15
# print(train_df.head(10))

# plot acoustic data distribution (0.5% of overall data) (Figure 1)
acoustic_sample1 = train_df['acoustic_data'].values[::200]
time_to_failure_sample1 = train_df['time_to_failure'].values[::200]
title1 = "Acoustic data distribution"
data_label1 = "acoustic data (0.5% of overall data)"
plot_data_fields(acoustic_sample1, title1, data_label1)

# plot acoustic data distribution with values between -20 and 20 (Figure 2)
acoustic_sample2 = acoustic_sample1[(acoustic_sample1 < 21) & (acoustic_sample1 > -21)]
data_label2 = "acoustic data between -20 and 20 (0.5% of overall data)"
plot_data_fields(acoustic_sample2, title1, data_label2)

# plot time to failure data distribution (0.5% of overall data) (Figure 3)
title2 = "Time to failure distribution"
data_label3 = "time to failure data (0.5% of overall data)"
plot_data_fields(time_to_failure_sample1, title2, data_label3)

# plot 0.5% of data by sampling every 200 data points (Figure 4)
sample1 = "0.5% of overall data"
visualize_experimental_data(acoustic_sample1, time_to_failure_sample1, sample1)

# plot first 2% of data (Figure 5)
acoustic_sample3 = train_df['acoustic_data'].values[: int(np.floor((number_of_rows_train * 2) / 100))]
time_to_failure_sample2 = train_df['time_to_failure'].values[: int(np.floor((number_of_rows_train * 2) / 100))]
sample2 = "first 2% of overall data"
visualize_experimental_data(acoustic_sample3, time_to_failure_sample2, sample2)

