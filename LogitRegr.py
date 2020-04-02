from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from scipy import interp


def sep_X_y(df, X_cols, y_cols):
    return df.loc[:,X_cols].values, df.loc[:,y_cols].values


def get_data():
    train = pd.read_csv(r'data/train.csv', parse_dates=[0], dtype={'tag': 'int64'})
    test = pd.read_csv(r'data/test.csv', parse_dates=[0], dtype={'tag': 'int64'})
    X_cols = ['environment_tmp', 'moto_tmp_mean', 'power', 'wind_speed', 'generator_speed', 'wind_direction',
              'tmp_diff', 'pitch_angle_sd', 'acc_x', 'acc_y', 'pitch1_ng5_tmp', 'pitch2_ng5_DC',
              'pitch3_ng5_tmp', 'wind_direction_mean', 'pitch1_ng5_DC', 'pitch2_ng5_tmp', 'yaw_position',
              'pitch_angle_mean', 'yaw_speed', 'pitch3_ng5_DC', 'cp']
    y_col = 'tag'
    train_X, train_y = sep_X_y(train, X_cols, y_col)
    test_X, test_y = sep_X_y(test, X_cols, y_col)
    return train_X, train_y, test_X, test_y


train_X, train_y, test_X, test_y = get_data()

# create cross-validation instance
cv = StratifiedKFold(n_splits=5)
# create model
clf = LogisticRegression(solver='sag',
                         max_iter=500,
                         class_weight='balanced',
                         random_state=0).fit(train_X, train_y)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
for i, (train, valid) in enumerate(cv.split(train_X, train_y)):
    clf.fit(train_X[train], train_y[train])
    viz = metrics.plot_roc_curve(clf, train_X[valid], train_y[valid],
                         name='ROC fold {}'.format(i),
                         alpha=0.3, lw=1, ax=ax)
    interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

viz = metrics.plot_roc_curve(clf, test_X, test_y,
                         name='ROC test',
                         alpha=0.8, lw=1, color='green', ax=ax)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = metrics.auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC of Train (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)


std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="ROC Curve")
ax.legend(loc="lower right")
plt.show()

# test_score = clf.score(test_X, test_y)
# print('The score in test data is ' + str(test_score))

