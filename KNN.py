from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from collections import Counter
import datetime
from joblib import dump, load


class KNN(object):
    def __init__(self, clf, train, test, lr_features, targets, cv, model_name):
        """
        the construction method
        :param clf: classifier model_lr
        :param train: train data - dataframe
        :param test: test data - dataframe
        :param lr_features: features for LogitReg
        :param targets: y columns - list
        :param cv: number of cv - int
        :param rules: rules for prediction - string
        """
        self.clf = clf
        self.train = train
        self.test = test
        self.lr_features = lr_features
        self.targets = targets
        self.cv = StratifiedKFold(n_splits=cv)
        self.model_name=model_name
        self.train_X = train.loc[:, lr_features].values
        self.train_y = train.loc[:, targets].values

    # split feature and target
    def split_X_y(self, df, X_cols, y_cols):
        return df.loc[:, X_cols].values, df.loc[:, y_cols].values

    # data preparation
    def get_data(self):
        sample_size = len(self.train.loc[lambda df: df[self.targets]==1,:])
        train_sample = self.train.loc[lambda df: df[self.targets]==1,:].append(self.train.loc[lambda df: df[self.targets]==0,:].sample(sample_size))
        train_sample_X, train_sample_y = self.split_X_y(train_sample, self.lr_features, self.targets)
        test_X, test_y = self.split_X_y(self.test, self.lr_features, self.targets)
        return train_sample_X, train_sample_y, test_X, test_y

    def train_model(self):
        """
        train and estimate model_lr
        draw ROC curves of train and test
        :param clf: the model_lr for classifier
        :param cv: train data splited by StratifiedKFold
        :return: the trained model_lr
        """
        train_sample_X, train_sample_y, test_X, test_y = self.get_data()
        cv_data = self.cv.split(train_sample_X, train_sample_y)

        tprs = []  # list for saving TP rates in each cv
        aucs = []  # list for saving aucs in each cv
        mean_fpr = np.linspace(0, 1, 100)  # mean FP rates
        fig, ax = plt.subplots()
        for i, (train, valid) in enumerate(cv_data): # 5 fold training of model_lr
            self.clf.fit(train_sample_X[train], train_sample_y[train])
            # plot ROC
            viz = metrics.plot_roc_curve(self.clf, train_sample_X[valid], train_sample_y[valid], name='ROC fold {}'.format(i), alpha=0.3, lw=1, ax=ax)
            interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)  # get TP rates
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
        # plot ROC of test data
        metrics.plot_roc_curve(self.clf, test_X, test_y, name='ROC test', alpha=0.8, lw=1, color='green', ax=ax)
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
        plt.savefig('res_fig_knn/'+ self.model_name +'.png')
        print(r'5 Cross Validation Mean AUC: %0.2f, Standard Deviation is %0.2f' % (mean_auc, std_auc))
        # train with all train data and compute the train and test accuracy respectively
        start = datetime.datetime.now()
        self.clf.fit(train_sample_X, train_sample_y)
        test_pred_y = self.clf.predict(test_X)
        end = datetime.datetime.now()
        print('Fit Time:')
        print(end - start)
        dump(self.clf, 'model_knn/' + self.model_name + '.joblib')
        train_acc = self.clf.score(self.train_X, self.train_y)
        test_acc = self.clf.score(test_X, test_y)
        print('Train Accuracy is %0.2f, Test Accuracy is %0.2f' %(train_acc, test_acc))
        train_pred_y = self.clf.predict(self.train_X)
        return self.train_y, train_pred_y, test_y, test_pred_y


def rule(train, test, rules):
    """
    generate train_rule, test_rule, train, test
    :return:
    """
    train_rule = train.loc[lambda df: eval(rules), :]
    train.drop(index=train_rule.index, axis=0, inplace=True)
    test_rule = test.loc[lambda df: eval(rules), :]
    test.drop(index=test_rule.index, axis=0, inplace=True)
    return train_rule, test_rule, train, test


def competition_score(pred_y, actual_y):
    y_pred_actual = np.array([pred_y, actual_y]).T
    # TP:0, FN:1, FP: 2, TN:3
    y_pred_actual_count = Counter(np.dot(y_pred_actual, np.array([1, 2])))
    TP = y_pred_actual_count.get(0) if y_pred_actual_count.get(0) is not None else 0
    FN = y_pred_actual_count.get(1) if y_pred_actual_count.get(1) is not None else 0
    FP = y_pred_actual_count.get(2) if y_pred_actual_count.get(2) is not None else 0
    TN = y_pred_actual_count.get(3) if y_pred_actual_count.get(3) is not None else 0
    print('TP:%d, FN:%d, FP:%d, TN:%d' % (TP, FN, FP, TN))
    # normal: 0, ice:1
    y_actual_count = Counter(actual_y)
    print(y_actual_count)
    score = (1 - 0.5 * (FN/y_actual_count.get(0)) - 0.5 * (FP/y_actual_count.get(1))) * 100
    #print("The final test score is {:.2f}".format(score))
    return score


if __name__ == '__main__':
    lr_features = ['r_windspeed_to_power', 'environment_tmp', 'moto_tmp_mean', 'tmp_diff',
                    'pitch_angle_sd', 'moto_tmp_sd', 'pitch_angle_mean', 'wind_speed', 'wind_direction',
                    'generator_speed', 'r_windspeed_to_generator_speed', 'pitch1_ng5_tmp', 'r_square',
                    'pitch1_ng5_DC', 'yaw_speed', 'acc_x', 'power', 'pitch2_ng5_DC', 'pitch3_ng5_tmp',
                    'wind_direction_mean']

    features = ['generator_speed', 'power', 'environment_tmp', 'pitch3_ng5_tmp',
                'pitch1_ng5_DC', 'tmp_diff', 'r_windspeed_to_power',
                'r_windspeed_to_generator_speed', 'r_square', 'moto_tmp_mean']



    features_p1 =['wind_speed', 'generator_speed', 'power', 'environment_tmp', 'pitch1_ng5_tmp',
                    'tmp_diff', 'r_windspeed_to_power', 'r_windspeed_to_generator_speed', 'r_square',
                    'moto_tmp_mean']


    features_p2 = ['wind_speed', 'generator_speed', 'power', 'environment_tmp',
                    'pitch3_ng5_tmp', 'tmp_diff', 'r_windspeed_to_power',
                    'r_windspeed_to_generator_speed', 'r_square', 'moto_tmp_mean']


    target = 'tag'
    train = pd.read_csv(r'data/train1.csv', parse_dates=[0], dtype={'tag': 'int64'})
    test = pd.read_csv(r'data/test1.csv', parse_dates=[0], dtype={'tag': 'int64'})
    rules = "(df['wind_speed']<-2) | (df['wind_speed']>2) | (df['generator_speed']<-2) | " \
            "(df['environment_tmp']>2) | (df['pitch2_ng5_tmp']>2) | (df['cp']>30) | " \
            "(df['ct']<-10)"
    train_rule, test_rule, train, test = rule(train, test, rules)
    train_rule.loc[:, 'pred'] = 0
    train_pred_y = train_rule.loc[:, 'pred'].values
    train_actual_y = train_rule.loc[:, 'tag'].values
    test_rule.loc[:, 'pred'] = 0
    test_pred_y = test_rule.loc[:, 'pred'].values
    test_actual_y = test_rule.loc[:, 'tag'].values

    # for data has time series info
    """
    clf = load('model_lr/LR_Rule2.joblib')
    train_has_null = train[train.isnull().T.any()]  # take down
    if len(train_has_null) >= 1:
        train_has_null_pred_y = clf.predict(train_has_null.loc[:, lr_features].values)
        train_pred_y = np.hstack((train_pred_y, train_has_null_pred_y))
        train_actual_y = np.hstack((train_actual_y, train_has_null.loc[:, target].values))
        train.drop(index=train_has_null.index, axis=0, inplace=True)

    test_has_null = test[test.isnull().T.any()]
    if len(test_has_null) >= 1:
        test_has_null_pred_y = clf.predict(test_has_null.loc[:, lr_features].values)
        test_pred_y = np.hstack((test_pred_y, test_has_null_pred_y))
        test_actual_y = np.hstack((test_actual_y, test_has_null.loc[:, target].values))
        test.drop(index=test_has_null.index, axis=0, inplace=True)
    """

    clf = KNeighborsClassifier(n_neighbors=5, weights="uniform")
    # without data division
    """
    knn: KNN = KNN(clf, train, test, features, target, 5, 'KNN_Rule2_k5')
    train_y, train_model_pred_y, test_y, test_model_pred_y = knn.train_model()
    train_pred_y = np.hstack((train_pred_y, train_model_pred_y))
    train_actual_y = np.hstack((train_actual_y, train_y))
    train_competition_score = competition_score(train_pred_y, train_actual_y)
    print("The Train Competition Score is {:.2f}".format(train_competition_score))
    test_pred_y = np.hstack((test_pred_y, test_model_pred_y))
    test_actual_y = np.hstack((test_actual_y, test_y))
    test_competition_score = competition_score(test_pred_y, test_actual_y)
    print("The Test Competition Score is {:.2f}".format(test_competition_score))
    """

    # divide data by wind_speed = -0.5
    #"""
    train_part1 = train.loc[lambda df: df['wind_speed'] <= -0.5,:]
    train_part2 = train.loc[lambda df: df['wind_speed'] > -0.5, :]
    test_part1 = test.loc[lambda df: df['wind_speed'] <= -0.5,:]
    test_part2 = test.loc[lambda df: df['wind_speed'] > -0.5, :]
    knn1: KNN = KNN(clf, train_part1, test_part1, features_p1, target, 5, 'KNN_Rule_Division2_k5_part1')
    train_y1, train_pred_y1, test_y1, test_pred_y1 = knn1.train_model()
    knn2: KNN = KNN(clf, train_part2, test_part2, features_p2, target, 5, 'KNN_Rule_Division2_k5_part2')
    train_y2, train_pred_y2, test_y2, test_pred_y2 = knn2.train_model()
    # join part1 and part2
    train_model_pred_y = np.hstack((train_pred_y1, train_pred_y2))
    train_y = np.hstack((train_y1, train_y2))
    test_model_pred_y = np.hstack((test_pred_y1, test_pred_y2))
    test_y = np.hstack((test_y1, test_y2))
    # join model_lr res and rule res
    train_pred_y = np.hstack((train_pred_y, train_model_pred_y))
    train_actual_y = np.hstack((train_actual_y, train_y))
    test_pred_y = np.hstack((test_pred_y, test_model_pred_y))
    test_actual_y = np.hstack((test_actual_y, test_y))
    train_competition_score = competition_score(train_pred_y, train_actual_y)
    print("The Train Competition Score is {:.2f}".format(train_competition_score))
    test_competition_score = competition_score(test_pred_y, test_actual_y)
    print("The Test Competition Score is {:.2f}".format(test_competition_score))
    #"""

