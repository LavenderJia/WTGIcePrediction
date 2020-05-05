from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from collections import Counter
import datetime


class LogitRegir(object):
    def __init__(self, clf, train, test, lr_features, targets, cv, rules):
        """
        the construction method
        :param clf: classifier model
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
        self.rules = rules

    # split feature and target
    def split_X_y(self, df, X_cols, y_cols):
        return df.loc[:, X_cols].values, df.loc[:, y_cols].values

    # data preparation
    def get_data(self):
        train_X, train_y = self.split_X_y(self.train, self.lr_features, self.targets)
        test_X, test_y = self.split_X_y(self.test, self.lr_features, self.targets)
        return train_X, train_y, test_X, test_y

    def rule(self):
        """
        generate self.pred_y_rule, self.test_y_rule, new self.train, self.test
        :return:
        """
        self.train.loc[:, 'pred_y'] = -1
        self.train.loc[lambda df: eval(self.rules), 'pred_y'] = 0
        self.train = self.train.loc[lambda df: df['pred_y'] == -1, :]

        self.test.loc[:, 'pred_y'] = -1
        self.test.loc[lambda df: eval(self.rules), 'pred_y'] = 0
        self.pred_y_rule = self.test.loc[lambda df: df['pred_y']==0, 'pred_y'].values
        self.test_y_rule = self.test.loc[lambda df: df['pred_y']==0, 'tag'].values
        self.test = self.test.loc[lambda df: df['pred_y'] == -1, :]

    def train_model(self):
        """
        train and estimate model
        draw ROC curves of train and test
        :param clf: the model for classifier
        :param cv: train data splited by StratifiedKFold
        :return: the trained model
        """
        train_X, train_y, test_X, test_y = self.get_data()
        cv_data = self.cv.split(train_X, train_y)

        tprs = []  # list for saving TP rates in each cv
        aucs = []  # list for saving aucs in each cv
        mean_fpr = np.linspace(0, 1, 100)  # mean FP rates
        fig, ax = plt.subplots()
        for i, (train, valid) in enumerate(cv_data): # 5 fold training of model
            self.clf.fit(train_X[train], train_y[train])
            # plot ROC
            viz = metrics.plot_roc_curve(self.clf, train_X[valid], train_y[valid], name='ROC fold {}'.format(i), alpha=0.3, lw=1, ax=ax)
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
        plt.savefig('res_fig/LogitRegrRes.png')

        pred_y = self.clf.predict(test_X)
        return test_y, pred_y

    # calculate test score
    def test_score(self, pred_y, test_y):
        y_pred_test = np.array([pred_y, test_y]).T
        # TP:0, FN:1, FP: 2, TN:3
        y_pred_test_count = Counter(np.dot(y_pred_test, np.array([1, 2])))
        print(y_pred_test_count)
        # normal: 0, ice:1
        y_test_count = Counter(test_y)
        print(y_test_count)
        score = (1 - 0.5 * (y_pred_test_count.get(1)/y_test_count.get(0)) - 0.5 * (y_pred_test_count.get(2)/y_test_count.get(1))) * 100
        print("The final test score is {:.2f}".format(score))

    def run(self, rules=True):
        if rules:
            self.rule()
            test_y_lr, pred_y_lr = self.train_model()
            #print(len(test_y_lr))
            #print(pred_y_lr[0:10])
            pred_y = np.hstack((self.pred_y_rule, pred_y_lr))
            test_y = np.hstack((self.test_y_rule, test_y_lr))
            self.test_score(pred_y, test_y)
        else:
            test_y, pred_y = self.train_model()
            self.test_score(pred_y, test_y)


if __name__ == '__main__':
    start = datetime.datetime.now()
    clf = LogisticRegression(solver='sag', max_iter=500, class_weight='balanced', random_state=0)
    lr_features = ['wind_speed',
       'generator_speed', 'power',
       'acc_x', 'environment_tmp',
       'pitch1_ng5_tmp',  'pitch3_ng5_tmp', 'pitch1_ng5_DC',
       'pitch2_ng5_DC',  'tmp_diff',  'cp',
       'r_windspeed_to_power', 'r_windspeed_to_generator_speed', 'r_square',
       'pitch_angle_mean', 'pitch_angle_sd',
       'moto_tmp_mean', 'moto_tmp_sd']
    target = 'tag'
    train = pd.read_csv(r'data/train.csv', parse_dates=[0], dtype={'tag': 'int64'})
    test = pd.read_csv(r'data/test.csv', parse_dates=[0], dtype={'tag': 'int64'})
    rules = "(df['wind_speed']<-2) | (df['wind_speed']>2) | (df['generator_speed']<-2) | " \
            "(df['environment_tmp']>2) | (df['pitch2_ng5_tmp']>2) | (df['cp']>30) | " \
            "(df['ct']<-10)"
    lr: LogitRegir = LogitRegir(clf, train, test, lr_features, target, 5, rules)
    lr.run(rules=False)
    end = datetime.datetime.now()
    print(end - start)
