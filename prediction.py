import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgbm
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from feature_generation_temp import preprocessing


def train_model(X, y, X_test):
    oof = np.array([0.0] * X.shape[0])
    prediction = np.array([0.0] * X_test.shape[0])
    scores = []
    feature_importance = pd.DataFrame()

    for train_group, test_group in KFold(n_splits=3, shuffle=True, random_state=11).split(X):
        X_train, X_valid = X.iloc[train_group], X.iloc[test_group]
        y_train, y_valid = y.iloc[train_group], y.iloc[test_group]

        feature_model = lgbm.LGBMRegressor(
            n_estimators=50000,
            n_jobs=-1,
            num_leaves=128,
            min_child_samples=79,
            objective='gamma',
            max_depth=-1,
            learning_rate=0.01,
            boosting_type='gbdt',
            subsample_freq=5,
            subsample=0.8126672064208567,
            bagging_seed=11,
            metric='mae',
            verbosity=-1,
            reg_alpha=0.1302650970728192,
            reg_lambda=0.3603427518866501,
            colsample_bytree=0.2
        )

        feature_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae', verbose=10000, early_stopping_rounds=200)

        y_pred_valid = feature_model.predict(X_valid)

        oof[test_group] = y_pred_valid.reshape(-1, )
        scores.append(mean_absolute_error(y_valid, y_pred_valid))

        prediction += feature_model.predict(X_test, num_iteration=feature_model.best_iteration_)

        # feature importance
        fold_importance = pd.DataFrame()
        fold_importance["feature"] = X.columns
        fold_importance["importance"] = feature_model.feature_importances_
        fold_importance["fold"] = 4   # no of folds + 1
        feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= 3   # no of folds

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    feature_importance["importance"] /= 3   # no of folds
    cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:50].index

    best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

    print(best_features)

    # plt.figure(figsize=(16, 12))
    # sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    # plt.title('LGB Features (avg over folds)')
    # plt.show()

    return oof, prediction, feature_importance


def predict(xtrain, ytrain, xtest):
    oof_lgb, prediction_lgb, feature_importance = train_model(xtrain, ytrain, xtest)

    return oof_lgb, prediction_lgb, feature_importance


if __name__ == '__main__':
    file_path = os.getcwd() + '/data_files/'

    X_train, y_train, X_test = preprocessing(file_path)
    oof_lgb, prediction_lgb, feature_importance = predict(X_train, y_train, X_test)

    # plt.figure(figsize=(18, 8))
    # plt.plot(y_train, color='g', label='y_train')
    # plt.plot(oof_lgb, color='b', label='lgb')
    # plt.legend(loc=(1, 0.5))
    # plt.title('lgb')
    # plt.show()

    # submission = pd.read_csv(file_path + 'sample_submission.csv', index_col='seg_id')
    # submission['time_to_failure'] = prediction_lgb
    # print(submission.head())
    # submission.to_csv('submission.csv')
    #
    # X.to_csv('train_features.csv', index=False)
    # X_test.to_csv('test_features.csv', index=False)
    # pd.DataFrame(y).to_csv('y.csv', index=False)
