import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from scipy import interp
from collections import Counter


# split feature and target
def split_X_y(df, X_cols, y_cols):
    return df.loc[:,X_cols].values, df.loc[:,y_cols].values


# data preparation
def get_data():
    train = pd.read_csv(r'data/train.csv', parse_dates=[0], dtype={'tag': 'int64'})
    test = pd.read_csv(r'data/test.csv', parse_dates=[0], dtype={'tag': 'int64'})
    X_cols = ['environment_tmp', 'r_windspeed_to_power', 'r_square',
       'r_windspeed_to_generator_speed', 'pitch_angle_mean', 'moto_tmp_mean',
       'pitch_angle_sd', 'power', 'tmp_diff', 'generator_speed', 'wind_speed',
       'moto_tmp_sd', 'torque', 'pitch2_ng5_tmp', 'pitch1_ng5_tmp',
       'pitch3_ng5_tmp', 'wind_direction']
    y_col = 'tag'
    train_X, train_y = split_X_y(train, X_cols, y_col)
    test_X, test_y = split_X_y(test, X_cols, y_col)
    return train_X, train_y, test_X, test_y


# Parameter optimization
def search_best_param(model, train_X, train_y, test_X, test_y):
    clf = GridSearchCV(model,
                       {
                           #'n_estimators': [45, 50, 55, 60, 65],  # step 1
                           # 'max_depth': [18, 20, 22, 24],  # step 2
                           #'min_child_weight': [1, 2, 3, 4, 5],  # step 2
                           # 'gamma': [0.1, 0.2, 0.3],  # step 3
                           # 'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9],  # step 4
                           # 'reg_alpha': [2, 3, 4, 5], 'reg_lambda': [2, 3, 4, 5],
                           'scale_pos_weight': [ 26, 28, 30, 32, 34]
                       })
    clf.fit(train_X, train_y, early_stopping_rounds=10,
            eval_set=[
                (train_X, train_y),
                (test_X, test_y)
            ],
            eval_metric='auc', verbose=True)
    print(clf.best_score_)
    print(clf.best_params_)


# model training
def train_model(clf, cv_data, test_X, test_y):
    """
    train and estimate model
    draw ROC curves of train and test
    :param clf: the model for classifier
    :param cv: train data splited by StratifiedKFold
    :return: the trained model
    """

    tprs = []  # list for saving TP rates in each cv
    aucs = []  # list for saving aucs in each cv
    mean_fpr = np.linspace(0, 1, 100)  # mean FP rates
    fig, ax = plt.subplots()
    for i, (train, valid) in enumerate(cv_data): # 5 fold training of model
        clf.fit(train_X[train], train_y[train],early_stopping_rounds=10,
                eval_set=[(train_X, train_y)],
                eval_metric='auc', verbose=True)
        # plot ROC
        viz = metrics.plot_roc_curve(clf, train_X[valid], train_y[valid], name='ROC fold {}'.format(i), alpha=0.3, lw=1, ax=ax)
        interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)  # get TP rates
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    # plot ROC of test data
    metrics.plot_roc_curve(clf, test_X, test_y, name='ROC test', alpha=0.8, lw=1, color='green', ax=ax)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    # draw mean auc of 5 cv train
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC of Train (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    # draw confident interval
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="ROC Curve")
    ax.legend(loc="lower right")
    plt.savefig('res_fig/XGBoostRes.png')
    return clf


# calculate test score
def test_score(pred_y, test_y):
    y_pred_test = np.array([pred_y, test_y]).T
    # TP:0, FN:1, FP: 2, TN:3
    y_pred_test_count = Counter(np.dot(y_pred_test, np.array([1,2])))
    # normal: 0, ice:1
    y_test_count = Counter(test_y)
    score = (1 - 0.5 * (y_pred_test_count.get(1)/y_test_count.get(0)) - 0.5 * (y_pred_test_count.get(2)/y_test_count.get(1))) * 100
    print("The final test score is {:.2f}".format(score))


if __name__ == '__main__':
    train_X, train_y, test_X, test_y = get_data()
    # create cross-validation instance
    cv = StratifiedKFold(n_splits=5)
    cv_data = cv.split(train_X, train_y)

    params = {
        'booster': 'gbtree',
        'objective': 'reg:logistic',  # 二分类的问题
        'n_estimators': 55,
        # 'num_class': 2,  # 类别数，与 multisoftmax 并用
        'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 20,  # 构建树的深度，越大越容易过拟合
        'eval_metric': 'auc',
        'scale_pos_weight': 28, # balance categories
        'reg_lambda': 3,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'reg_alpha': 5,
        'subsample': 1,  # 随机采样训练样本
        'colsample_bytree': 0.6,  # 生成树时进行的列采样
        'min_child_weight': 2,
        'verbosity': 2,  # 设置成1则没有运行信息输出，最好是设置为0.
        'learning_rate': 0.007,  # 如同学习率
        'random_state': 1000,
        'nthread': 4,  # cpu 线程数
    }
    clf = xgb.XGBClassifier(**params)
    # search_best_param(clf, train_X, train_y, test_X, test_y)
    clf = train_model(clf, cv_data, test_X, test_y)
    pred_y = clf.predict(test_X)
    test_score(pred_y, test_y)
