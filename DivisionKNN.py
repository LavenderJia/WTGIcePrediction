# 导入模块
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from collections import Counter

wtg_15 = pd.read_csv(r'data/train.csv', parse_dates=[0], dtype={'tag': 'int64'})
wtg_15 = wtg_15.drop(['time', 'rec_time_interval', 'yaw_position', 'yaw_speed', 'acc_y', 'int_tmp',
                           'pitch3_ng5_DC', 'pitch_angle_mean', 'pitch_angle_sd', 'pitch_speed_mean',
                           'pitch_speed_sd', 'moto_tmp_sd', 'diff_pitch_angle', 'diff_moto_tmp',
                           'diff_pitch1_ng5_tmp', 'diff_pitch2_ng5_tmp', 'diff_pitch3_ng5_tmp'], axis=1)

wtg_21 = pd.read_csv(r'data/test.csv', parse_dates=[0], dtype={'tag': 'int64'})
wtg_21 = wtg_21.drop(['time', 'rec_time_interval', 'yaw_position', 'yaw_speed', 'acc_y', 'int_tmp',
                           'pitch3_ng5_DC', 'pitch_angle_mean', 'pitch_angle_sd', 'pitch_speed_mean',
                           'pitch_speed_sd', 'moto_tmp_sd', 'diff_pitch_angle', 'diff_moto_tmp',
                           'diff_pitch1_ng5_tmp', 'diff_pitch2_ng5_tmp', 'diff_pitch3_ng5_tmp'], axis=1)  #  保留筛选后的特征

# 特征分割，根据风速与-0.5的大小，将数据分成两部分
wtg0 = wtg_15.loc[lambda df: df['wind_speed'] < -0.5, :]
wtg1 = wtg_15.loc[lambda df: df['wind_speed'] >= -0.5, :]

wtg0_21 = wtg_21.loc[lambda df: df['wind_speed'] < -0.5, :]
wtg1_21 = wtg_21.loc[lambda df: df['wind_speed'] >= -0.5, :]


def train_model(clf, train_X, train_y, test_X, test_y, part):
    """
    train and estimate model
    draw ROC curves of train and test
    :param clf: the model for classifier
    :param cv: train data splited by StratifiedKFold
    :return: the trained model
    """
    cv_data = StratifiedKFold(n_splits=5).split(train_X, train_y)

    tprs = []  # list for saving TP rates in each cv
    aucs = []  # list for saving aucs in each cv
    mean_fpr = np.linspace(0, 1, 100)  # mean FP rates
    fig, ax = plt.subplots()
    for i, (train, valid) in enumerate(cv_data):  # 5 fold training of model
        clf.fit(train_X[train], train_y[train])
        # plot ROC
        viz = metrics.plot_roc_curve(clf, train_X[valid], train_y[valid], name='ROC fold {}'.format(i), alpha=0.3,
                                     lw=1, ax=ax)
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
    plt.savefig('res_fig/KNNRes_part%d.png' % part)
    pred_y = clf.predict(test_X)
    return pred_y

    # calculate test score
def test_score(pred_y, test_y):
    y_pred_test = np.array([pred_y, test_y]).T
    # TP:0, FN:1, FP: 2, TN:3
    y_pred_test_count = Counter(np.dot(y_pred_test, np.array([1, 2])))
    print(y_pred_test_count)
    # normal: 0, ice:1
    y_test_count = Counter(test_y)
    print(y_test_count)
    score = (1 - 0.5 * (y_pred_test_count.get(1)/y_test_count.get(0)) - 0.5 * (y_pred_test_count.get(2)/y_test_count.get(1))) * 100
    print("The final test score is {:.2f}".format(score))

"""
# 用15号风机全部样本数据划分训练集和测试集，利用测试集查看算法预测效果的得分
wtg_15_new = wtg0.copy()
wtg_21_new = wtg0_21.copy()
wtg_15_new.drop(['tag'], axis=1, inplace=True)
wtg_21_new.drop(['tag'], axis=1, inplace=True)
x_train, x_test, y_train, test_y1 = wtg_15_new.values, wtg_21_new.values, wtg0.tag.values, wtg0_21.tag.values # 将数据按15号风机为训练集，21号风机作为测试集
clf = KNeighborsClassifier(n_neighbors=4, weights='distance', n_jobs=4)  # 这里k值由前面的最佳k值确定，weights参数选用'distance'，减小样本不均衡带来的影响
pred_y1 = train_model(clf, x_train, y_train, x_test, test_y1, 1)
#clf.fit(x_train, y_train)
#score = clf.score(x_test, y_test, sample_weight=None)  # 通过测试集判断KNN预测效果得分
#print('%.4f' % score)   # k=4  得分是0.9772


# 用全部样本数据划分训练集和测试集，利用测试集查看算法预测效果的得分
wtg_15_new = wtg1.copy()
wtg_21_new = wtg1_21.copy()
wtg_15_new.drop(['tag'], axis=1, inplace=True)
wtg_21_new.drop(['tag'], axis=1, inplace=True)
x_train, x_test, y_train, test_y2 = wtg_15_new.values, wtg_21_new.values, wtg1.tag.values, wtg1_21.tag.values # 将数据按15号风机为训练集，21号风机作为测试集
clf = KNeighborsClassifier(n_neighbors=4, weights='distance', n_jobs=4)  # 这里weights参数选用'distance'，有助于减小样本不均衡带来的影响
#clf.fit(x_train, y_train)
pred_y2 = train_model(clf, x_train, y_train, x_test, test_y2, 2)
pred_y = np.hstack((pred_y1, pred_y2))
test_y = np.hstack((test_y1, test_y2))
test_score(pred_y, test_y)
"""

# 不做特征分割
# 用全部样本数据划分训练集和测试集，利用测试集查看算法预测效果的得分
wtg_15_new = wtg_15.copy()
wtg_21_new = wtg_21.copy()
wtg_15_new.drop(['tag'], axis=1, inplace=True)
wtg_21_new.drop(['tag'], axis=1, inplace=True)
x_train, x_test, y_train, test_y = wtg_15_new.values, wtg_21_new.values, wtg_15.tag.values, wtg_21.tag.values # 将数据按15号风机为训练集，21号风机作为测试集
clf = KNeighborsClassifier(n_neighbors=4, weights='distance', n_jobs=4)  # 这里weights参数选用'distance'，有助于减小样本不均衡带来的影响
#clf.fit(x_train, y_train)
pred_y = train_model(clf, x_train, y_train, x_test, test_y, 3)
test_score(pred_y, test_y)


