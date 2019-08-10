import numpy as np
# import pandas as pd
import xgboost as xgb
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

warnings.simplefilter(action='ignore', category=FutureWarning)


def predict(xtrain, ytrain, xtest):
    oof = np.array([0.0] * xtrain.shape[0])
    predicted_val = np.array([0.0] * xtest.shape[0])
    scores = []
    # feature_importance = pd.DataFrame()

    for train_group, test_group in KFold(n_splits=5, shuffle=True, random_state=11).split(xtrain):
        xtrainkf, xtestkf, ytrainkf, ytestkf = xtrain.iloc[train_group], xtrain.iloc[test_group], ytrain.iloc[train_group], ytrain.iloc[test_group]

        train_data = xgb.DMatrix(data=xtrainkf, label=ytrainkf, feature_names=xtrain.columns)
        test_data = xgb.DMatrix(data=xtestkf, label=ytestkf, feature_names=xtrain.columns)

        params = {
            'eta': 0.03,
            'max_depth': 9,
            'subsample': 0.85,
            'colsample_bytree': 0.3,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'silent': True,
            'nthread': -1
        }

        feature_model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=[(train_data, 'train_data'), (test_data, 'valid_data')], early_stopping_rounds=200, verbose_eval=500, params=params)
        ykf_predicted = feature_model.predict(xgb.DMatrix(xtestkf, feature_names=xtrain.columns), ntree_limit=feature_model.best_ntree_limit)
        predicted_val += feature_model.predict(xgb.DMatrix(xtest, feature_names=xtrain.columns), ntree_limit=feature_model.best_ntree_limit)

        oof[test_group] = ykf_predicted.reshape(-1, )
        scores.append(mean_absolute_error(ytestkf, ykf_predicted))

        print(sorted(feature_model.get_score(importance_type='gain').items(), key=lambda v: v[1]))

        # # feature importance
        # fold_importance = pd.DataFrame()
        # fold_importance["feature"] = xtrain.columns
        # fold_importance["importance"] = feature_model.feature_importances_
        # # fold_importance["importance"] = feature_model.get_score(importance_type='gain')
        # fold_importance["fold"] = 6   # no of folds + 1
        # feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    predicted_val /= 5
    print(f'CV mean score: {np.mean(scores):.4f}, std: {np.std(scores):1.4f}.')

    # feature_importance["importance"] /= 5   # no of folds
    # cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:50].index
    #
    # best_features = feature_importance.loc[feature_importance.feature.isin(cols)]
    #
    # print(best_features)

    # plt.figure(figsize=(16, 12))
    # sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    # plt.title('LGB Features (avg over folds)')
    # plt.show()

    return oof, predicted_val
