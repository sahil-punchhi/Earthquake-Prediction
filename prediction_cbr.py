import numpy as np
from catboost import CatBoostRegressor
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

warnings.simplefilter(action='ignore', category=FutureWarning)


def predict(xtrain, ytrain, xtest):
    oof = np.array([0.0] * xtrain.shape[0])
    predicted_val = np.array([0.0] * xtest.shape[0])
    scores = []

    for train_group, test_group in KFold(n_splits=5, shuffle=True, random_state=11).split(xtrain):
        xtrainkf, xtestkf, ytrainkf, ytestkf = xtrain.iloc[train_group], xtrain.iloc[test_group], ytrain.iloc[train_group], ytrain.iloc[test_group]

        params = {
            'num_leaves': 128,
            'min_data_in_leaf': 79,
            'objective': 'gamma',
            'max_depth': -1,
            'learning_rate': 0.01,
            'boosting': 'gbdt',
            'bagging_freq': 5,
            'bagging_fraction': 0.8126672064208567,
            'bagging_seed': 11,
            'metric': 'mae',
            'verbosity': -1,
            'reg_alpha': 0.1302650970728192,
            'reg_lambda': 0.3603427518866501,
            'feature_fraction': 0.2
        }

        # feature_model = CatBoostRegressor(iterations=20000, eval_metric='MAE', **params)
        feature_model = CatBoostRegressor(iterations=20, eval_metric='MAE')
        feature_model.fit(xtrainkf, ytrainkf, eval_set=(xtestkf, ytestkf), cat_features=[], use_best_model=True, verbose=False)

        ykf_predicted = feature_model.predict(xtestkf)
        predicted_val += feature_model.predict(xtest)

        oof[test_group] = ykf_predicted.reshape(-1, )
        scores.append(mean_absolute_error(ytestkf, ykf_predicted))

    predicted_val /= 5
    print(f'CV mean score: {np.mean(scores):.4f}, std: {np.std(scores):1.4f}.')

    return oof, predicted_val
