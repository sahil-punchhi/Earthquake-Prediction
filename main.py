import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from prediction_cbr import predict as cbrpredict
from prediction_lgb import predict as lgbpredict
from prediction_xgb import predict as xgbpredict
from feature_generation import preprocessing
from data_exploration import data_exploration

warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == '__main__':
    file_path = os.getcwd() + '/data_files/'

    xtrain, ytrain, xtest, ti = preprocessing(file_path)
    # predicted_val, out_of_fold = xgbpredict(xtrain, ytrain, xtest)
    predicted_val, out_of_fold = lgbpredict(xtrain, ytrain, xtest)
    # predicted_val, out_of_fold = cbrpredict(xtrain, ytrain, xtest)

    # data_exploration(pd.read_csv(file_path + 'train.csv', index=False))

    plt.figure(figsize=(18, 8))
    plt.plot(ytrain, color='g', label='y_train')
    plt.plot(out_of_fold, color='b', label='y_predicted')
    plt.legend(loc=(1, 0.5))
    plt.title('Prediction')
    plt.show()

    result = pd.read_csv(file_path + 'sample_submission.csv', index_col='seg_id')
    result['time_to_failure'] = predicted_val
    result.to_csv(file_path + 'result/sample_submission.csv')

    xtrain.to_csv(file_path + 'result/train_features.csv', index=False)
    xtest.to_csv(file_path + 'result/test_features.csv', index=False)
    pd.DataFrame(ytrain).to_csv(file_path + 'result/time_to_failure.csv', index=False)

    tf = datetime.now()
    tdiff = (tf - ti).total_seconds()

    print(f'Time taken for execution: {int(tdiff // 60)}min{int(tdiff % 60)}sec')