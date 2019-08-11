import numpy as np
import pandas as pd
import statistics
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

warnings.simplefilter(action='ignore', category=FutureWarning)


# prediction function and computation of mean absolute error and feature importance scores for LightGBM model
def predict(xtrain, ytrain, xtest):
    out_of_fold = np.array([0.0] * xtrain.shape[0])
    predicted_val = np.array([0.0] * xtest.shape[0])
    scores = []
    temp = {}

    for i in xtrain.columns:
        temp[i] = []

    # 5-fold cross validation
    for train_group, test_group in KFold(n_splits=5, shuffle=True, random_state=42).split(xtrain):
        # further divide each split into training and test data for the fold
        xtrainkf, xtestkf, ytrainkf, ytestkf = xtrain.iloc[train_group], xtrain.iloc[test_group], ytrain.iloc[train_group], ytrain.iloc[test_group]

        # define model parameters
        feature_model = lgb.LGBMRegressor(
            bagging_seed=42,
            boosting_type='gbdt',
            colsample_bytree=0.2,
            learning_rate=0.01,
            max_depth=-1,
            metric='mae',
            min_child_samples=60,
            n_estimators=50000,
            n_jobs=-1,
            num_leaves=125,
            objective='gamma',
            reg_alpha=0.130265,
            reg_lambda=0.360343,
            subsample=0.812667,
            subsample_freq=5,
            verbosity=-1
        )

        # fit model and predict
        feature_model.fit(xtrainkf, ytrainkf, eval_set=[(xtrainkf, ytrainkf), (xtestkf, ytestkf)], eval_metric='mae', verbose=10000, early_stopping_rounds=200)
        ykf_predicted = feature_model.predict(xtestkf)
        predicted_val += feature_model.predict(xtest, num_iteration=feature_model.best_iteration_)

        # get out_of_fold and score values for the fold
        out_of_fold[test_group] = ykf_predicted.reshape(-1, )
        scores.append(mean_absolute_error(ytestkf, ykf_predicted))

        # get feature importance scores for the fold
        ctr = 0
        for key, val in temp.items():
            temp[key].append(feature_model.feature_importances_[ctr])
            ctr += 1

    # compute mean of features over 5 folds
    fi = {}
    for key, val in temp.items():
        fi[key] = statistics.mean(val)

    feature_importances = pd.DataFrame(list(sorted(fi.items(), key=lambda v: v[1], reverse=True)), columns=['features', 'importance_score'])

    # sort features by importance and generate bar plot for best 50 features
    plt.figure(figsize=(16, 12))
    sns.barplot(x='importance_score', y='features', data=feature_importances.head(50))
    plt.title('LightGBM Feature Importances')
    plt.show()

    # print mean and standard deviation scores for model
    print(f'CV mean score: {np.mean(scores):.4f}, std: {np.std(scores):1.4f}.')

    return predicted_val / 5, out_of_fold
