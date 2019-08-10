import numpy as np
import pandas as pd
import statistics
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

warnings.simplefilter(action='ignore', category=FutureWarning)


def predict(xtrain, ytrain, xtest):
    out_of_fold = np.array([0.0] * xtrain.shape[0])
    predicted_val = np.array([0.0] * xtest.shape[0])
    scores = []
    temp = {}

    for i in xtrain.columns:
        temp[i] = []

    for train_group, test_group in KFold(n_splits=5, shuffle=True, random_state=11).split(xtrain):
        xtrainkf, xtestkf, ytrainkf, ytestkf = xtrain.iloc[train_group], xtrain.iloc[test_group], ytrain.iloc[train_group], ytrain.iloc[test_group]

        train_data = xgb.DMatrix(data=xtrainkf, label=ytrainkf, feature_names=xtrain.columns)
        test_data = xgb.DMatrix(data=xtestkf, label=ytestkf, feature_names=xtrain.columns)

        params = {
            'colsample_bytree': 0.3,
            'eta': 0.03,
            'eval_metric': 'mae',
            'max_depth': 9,
            'n_jobs': -1,
            'objective': 'reg:linear',
            'subsample': 0.85,
            'verbosity': 0
        }

        feature_model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=[(train_data, 'train_data'), (test_data, 'valid_data')], early_stopping_rounds=200, verbose_eval=500, params=params)
        ykf_predicted = feature_model.predict(xgb.DMatrix(xtestkf, feature_names=xtrain.columns), ntree_limit=feature_model.best_ntree_limit)
        predicted_val += feature_model.predict(xgb.DMatrix(xtest, feature_names=xtrain.columns), ntree_limit=feature_model.best_ntree_limit)

        out_of_fold[test_group] = ykf_predicted.reshape(-1, )
        scores.append(mean_absolute_error(ytestkf, ykf_predicted))

        # feature importance scores
        for key, val in feature_model.get_score(importance_type='gain').items():
            temp[key].append(val)

    fi = {}
    for key, val in temp.items():
        if val:
            fi[key] = statistics.mean(val)

    feature_importances = pd.DataFrame(list(sorted(fi.items(), key=lambda v: v[1], reverse=True)), columns=['features', 'importance_score'])

    plt.figure(figsize=(16, 12))
    sns.barplot(x='importance_score', y='features', data=feature_importances.head(50))
    plt.title('XGBoost Feature Importances')
    plt.show()

    print(f'CV mean score: {np.mean(scores):.4f}, std: {np.std(scores):1.4f}.')

    return (predicted_val / 5), out_of_fold
