import numpy as np
import pandas as pd
import statistics
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
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

        params = {
            'iterations': 20,
            'eval_metric': 'MAE'
        }

        feature_model = CatBoostRegressor(**params)
        feature_model.fit(xtrainkf, ytrainkf, eval_set=(xtestkf, ytestkf), cat_features=[], use_best_model=True, verbose=False)

        ykf_predicted = feature_model.predict(xtestkf)
        predicted_val += feature_model.predict(xtest)

        out_of_fold[test_group] = ykf_predicted.reshape(-1, )
        scores.append(mean_absolute_error(ytestkf, ykf_predicted))

        # feature importance scores
        ctr = 0
        for key, val in temp.items():
            temp[key].append(feature_model.feature_importances_[ctr])
            ctr += 1

    fi = {}
    for key, val in temp.items():
        fi[key] = statistics.mean(val)

    feature_importances = pd.DataFrame(list(sorted(fi.items(), key=lambda v: v[1], reverse=True)), columns=['features', 'importance_score'])

    plt.figure(figsize=(16, 12))
    sns.barplot(x='importance_score', y='features', data=feature_importances.head(50))
    plt.title('CatBoost Feature Importances')
    plt.show()

    print(f'CV mean score: {np.mean(scores):.4f}, std: {np.std(scores):1.4f}.')

    return (predicted_val / 5), out_of_fold
