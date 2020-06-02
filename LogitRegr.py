from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from collections import Counter
import datetime
from joblib import dump, load

# logistic regression class
class LogitRegir(object):
    def __init__(self, clf, train, test, lr_features, targets, cv, model_name):
        """
        the construction method
        :param clf: classifier model_lr
        :param train: train data - dataframe
        :param test: test data - dataframe
        :param lr_features: features for LogitReg
        :param targets: y columns - list
        :param cv: number of cv - int
        :param model_name: name of the model - string
        """
        self.clf = clf
        self.train = train
        self.test = test
        self.lr_features = lr_features
        self.targets = targets
        self.cv = StratifiedKFold(n_splits=cv)
        self.model_name=model_name

    # split feature and target
    def split_X_y(self, df, X_cols, y_cols):
        return df.loc[:, X_cols].values, df.loc[:, y_cols].values

    # function for spliting features and target
    def get_data(self):
        train_X, train_y = self.split_X_y(self.train, self.lr_features, self.targets)
        test_X, test_y = self.split_X_y(self.test, self.lr_features, self.targets)
        return train_X, train_y, test_X, test_y

    def train_model(self):
        """
        train and estimate model
        draw ROC curves of train and test
        :return: train_y, train_pred_y, test_y, test_pred_y
        """
        train_X, train_y, test_X, test_y = self.get_data()  # generate formed train and test data
        cv_data = self.cv.split(train_X, train_y)  # split train data to train and validation data

        tprs = []  # list for saving TP rates in each cv
        aucs = []  # list for saving aucs in each cv
        mean_fpr = np.linspace(0, 1, 100)  # mean FP rates
        fig, ax = plt.subplots()  # initialize plt
        for i, (train, valid) in enumerate(cv_data):  # 5 fold training of model_lr
            self.clf.fit(train_X[train], train_y[train])  # fit model using train data
            # plot ROC
            viz = metrics.plot_roc_curve(self.clf, train_X[valid], train_y[valid], name='ROC fold {}'.format(i), alpha=0.3, lw=1, ax=ax)
            interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)  # get TP rates and do interp
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)  # add new interp_tpr to trprs list
            aucs.append(viz.roc_auc)  # add viz.roc_auc to aucs list
        # plot ROC of test data
        metrics.plot_roc_curve(self.clf, test_X, test_y, name='ROC test', alpha=0.8, lw=1, color='green', ax=ax)
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
        # plot mean auc of 5 cv train
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC of Train (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)
        # draw confident interval
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)  # get upper bound
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)  # get lower bound
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title="ROC Curve")
        ax.legend(loc="lower right")
        plt.savefig('res_fig_lr/'+ self.model_name +'.png')
        print(r'5 Cross Validation Mean AUC: %0.2f, Standard Deviation is %0.2f' % (mean_auc, std_auc))
        # train with all train data and compute the train and test accuracy respectively
        start = datetime.datetime.now()  # record start time
        self.clf.fit(train_X, train_y)  # fit all train data
        test_pred_y = self.clf.predict(test_X)  # predict test data
        end = datetime.datetime.now()  # record end time
        print('Fit Time:')  # calculate time cost
        print(end - start)
        train_pred_y = self.clf.predict(train_X)  # train data prediction
        dump(self.clf, 'model_lr/' + self.model_name + '.joblib')  # save trained model
        train_acc = self.clf.score(train_X, train_y)  # calculate train accuracy
        test_acc = self.clf.score(test_X, test_y)  # calculate test accuracy
        print('Train Accuracy is %0.2f, Test Accuracy is %0.2f' %(train_acc, test_acc))
        return train_y, train_pred_y, test_y, test_pred_y


def rule(train, test, rules):
    """
    generate train_rule, test_rule, train, test
    :param train: train data - Dataframe
    :param test: test data - Dataframe
    :param rules: strong rules - string
    :return: train_rule, test_rule, train, test
    """
    train_rule = train.loc[lambda df: eval(rules), :]  # filter train data according to strong rules
    train.drop(index=train_rule.index, axis=0, inplace=True)
    test_rule = test.loc[lambda df: eval(rules), :]  # filter test data according to strong rules
    test.drop(index=test_rule.index, axis=0, inplace=True)
    return train_rule, test_rule, train, test


def competition_score(pred_y, actual_y):
    """
    calculate competition score
    :param pred_y: predicted y - np.ndarray
    :param actual_y: actual y - np.ndarray
    :return: score - float
    """
    y_pred_actual = np.array([pred_y, actual_y]).T  # concat actual and pred
    # TP:0, FN:1, FP: 2, TN:3
    y_pred_actual_count = Counter(np.dot(y_pred_actual, np.array([1, 2])))  # get confusion matrix
    print('TP:%d, FN:%d, FP:%d, TN:%d' % (y_pred_actual_count.get(0), y_pred_actual_count.get(1),
                                          y_pred_actual_count.get(2), y_pred_actual_count.get(3)))
    # normal: 0, ice:1
    y_actual_count = Counter(actual_y)
    print(y_actual_count)
    # calculate score according to formula in competition document
    score = (1 - 0.5 * (y_pred_actual_count.get(1)/y_actual_count.get(0)) - 0.5 * (y_pred_actual_count.get(2)/y_actual_count.get(1))) * 100
    #print("The final test score is {:.2f}".format(score))
    return score


if __name__ == '__main__':
    # features without time series info
    lr_features =['r_windspeed_to_power', 'environment_tmp', 'moto_tmp_mean', 'tmp_diff',
                    'pitch_angle_sd', 'moto_tmp_sd', 'pitch_angle_mean', 'wind_speed', 'wind_direction',
                    'generator_speed', 'r_windspeed_to_generator_speed', 'pitch1_ng5_tmp', 'r_square',
                    'pitch1_ng5_DC', 'yaw_speed', 'acc_x', 'power', 'pitch2_ng5_DC', 'pitch3_ng5_tmp',
                    'wind_direction_mean']
    # features with timeseries info
    lr_features_v2 = ['r_windspeed_to_power', 'environment_tmp', 'environment_tmp_lag560',
                        'wind_speed_lag560', 'tmp_diff', 'pitch_angle_sd', 'wind_speed_lag160',
                        'pitch_angle_mean', 'moto_tmp_sd', 'wind_speed_lag40', 'environment_tmp_lag400',
                        'wind_direction', 'pitch1_ng5_tmp', 'wind_speed', 'generator_speed',
                        'r_windspeed_to_generator_speed', 'r_square', 'environment_tmp_lag160',
                        'pitch1_ng5_DC', 'wind_speed_lag80', 'environment_tmp_lag240', 'acc_x',
                        'wind_speed_lag240', 'wind_speed_lag400', 'pitch3_ng5_tmp', 'pitch2_ng5_DC',
                        'wind_direction_mean']

    target = 'tag'
    train = pd.read_csv(r'data/train2.csv', parse_dates=[0], dtype={'tag': 'int64'})  # read train and test data
    test = pd.read_csv(r'data/test2.csv', parse_dates=[0], dtype={'tag': 'int64'})
    # define rules
    rules = "(df['wind_speed']<-2) | (df['wind_speed']>2) | (df['generator_speed']<-2) | " \
            "(df['environment_tmp']>2) | (df['pitch2_ng5_tmp']>2) | (df['cp']>30) | " \
            "(df['ct']<-10)"

    # add rules
    """
    train_rule, test_rule, train, test = rule(train, test, rules)  # filter train and test data by rules
    train_rule.loc[:, 'pred'] = 0  # set rule prediction of train data result 0
    train_pred_y = train_rule.loc[:, 'pred'].values  # get train rule prediction y
    train_actual_y = train_rule.loc[:, 'tag'].values  # get train rule actual y
    test_rule.loc[:, 'pred'] = 0  # set rule prediction of test data result 0
    test_pred_y = test_rule.loc[:, 'pred'].values  # get test rule prediction y
    test_actual_y = test_rule.loc[:, 'tag'].values  # get test rule prediction y
    """

    # for data has time series info
    """
    clf = load('model_lr/LR_Rule2.joblib')  # load model trained on data without time series info
    train_has_null = train[train.isnull().T.any()]  # get null data in train
    if len(train_has_null) >= 1:  # if train data has null in time series columns
        # use basic model to predict data has null in time series info
        train_has_null_pred_y = clf.predict(train_has_null.loc[:, lr_features].values)
        # concat result with previous step
        train_pred_y = np.hstack((train_pred_y, train_has_null_pred_y))
        train_actual_y = np.hstack((train_actual_y, train_has_null.loc[:, target].values))
        train.drop(index=train_has_null.index, axis=0, inplace=True)  # filter predicted train data

    test_has_null = test[test.isnull().T.any()]  # get null data in test
    if len(test_has_null) >= 1:  # if test data has null in time series columns
        # use basic model to predict data has null in time series info
        test_has_null_pred_y = clf.predict(test_has_null.loc[:, lr_features].values)
        # concat result with previous step
        test_pred_y = np.hstack((test_pred_y, test_has_null_pred_y))
        test_actual_y = np.hstack((test_actual_y, test_has_null.loc[:, target].values))
        test.drop(index=test_has_null.index, axis=0, inplace=True)  # filter predicted test data
    """
    # define logistic regression model
    clf = LogisticRegression(solver='sag', max_iter=500, class_weight='balanced', random_state=0)
    # without data division
    """
    # get instance of LogitRegir class
    lr: LogitRegir = LogitRegir(clf, train, test, lr_features, target, 5, 'LR_Rule_TSInfo2')
    train_y, train_model_pred_y, test_y, test_model_pred_y = lr.train_model()  # get model result
    # concat result with previous step
    train_pred_y = np.hstack((train_pred_y, train_model_pred_y))
    train_actual_y = np.hstack((train_actual_y, train_y))
    # get competition score of train data
    train_competition_score = competition_score(train_pred_y, train_actual_y)
    print("The Train Competition Score is {:.2f}".format(train_competition_score))
    # concat result with previous step
    test_pred_y = np.hstack((test_pred_y, test_model_pred_y))
    test_actual_y = np.hstack((test_actual_y, test_y))
    # get competition score of test data
    test_competition_score = competition_score(test_pred_y, test_actual_y)
    print("The Test Competition Score is {:.2f}".format(test_competition_score))
    """

    # no rules
    #"""
    lr: LogitRegir = LogitRegir(clf, train, test, lr_features, target, 5, 'LR_basic')
    train_y, train_model_pred_y, test_y, test_model_pred_y = lr.train_model()  # get model result
    train_competition_score = competition_score(train_model_pred_y, train_y)  # get competition score of train data
    print("The Train Competition Score is {:.2f}".format(train_competition_score))
    test_competition_score = competition_score(test_model_pred_y, test_y)  # get competition score of test data
    print("The Test Competition Score is {:.2f}".format(test_competition_score))
    #"""


    # divide data by wind_speed = -0.5
    """
    train_part1 = train.loc[lambda df: df['wind_speed'] <= -0.5,:]
    train_part2 = train.loc[lambda df: df['wind_speed'] > -0.5, :]
    test_part1 = test.loc[lambda df: df['wind_speed'] <= -0.5,:]
    test_part2 = test.loc[lambda df: df['wind_speed'] > -0.5, :]
    # get instance of LogitRegir class for low wind speed data
    lr1: LogitRegir = LogitRegir(clf, train_part1, test_part1, lr_features_v2, target, 5, 'LR_Rule_TSInfo_Division2_part1')
    train_y1, train_pred_y1, test_y1, test_pred_y1 = lr1.train_model()  # get model result for low wind speed data
    # get instance of LogitRegir class for high wind speed data
    lr2: LogitRegir = LogitRegir(clf, train_part2, test_part2, lr_features_v2, target, 5, 'LR_Rule_TSInfo_Division2_part2')
    train_y2, train_pred_y2, test_y2, test_pred_y2 = lr2.train_model()   # get model result for high wind speed data
    # concat part1 and part2 result
    train_model_pred_y = np.hstack((train_pred_y1, train_pred_y2))
    train_y = np.hstack((train_y1, train_y2))
    test_model_pred_y = np.hstack((test_pred_y1, test_pred_y2))
    test_y = np.hstack((test_y1, test_y2))
    # concat result with previous step
    train_pred_y = np.hstack((train_pred_y, train_model_pred_y))
    train_actual_y = np.hstack((train_actual_y, train_y))
    test_pred_y = np.hstack((test_pred_y, test_model_pred_y))
    test_actual_y = np.hstack((test_actual_y, test_y))
    train_competition_score = competition_score(train_pred_y, train_actual_y)  # get competition score of train data
    print("The Train Competition Score is {:.2f}".format(train_competition_score))
    test_competition_score = competition_score(test_pred_y, test_actual_y)  # get competition score of test data
    print("The Test Competition Score is {:.2f}".format(test_competition_score))
    """

