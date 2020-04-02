import pandas as pd
"""
This code is for tag matching,
dealing with both 15 and 21 wtg.
"""


# change the directory here to your own directory
# and also remember to change the diretory in the end of this code
def read_data(wtg_num):
    directory = r'E:/Docs/Jupyter/WTG_Ice_Prediction/Data/' + str(wtg_num) + r'/' + str(wtg_num)
    wtg = pd.read_csv(directory + r'_data.csv', parse_dates=[0])
    wtg_normalInfo = pd.read_csv(directory + r'_normalInfo.csv', parse_dates=[0,1])
    wtg_failureInfo = pd.read_csv(directory + r'_failureInfo.csv', parse_dates=[0,1])
    print('data of wtg ' + str(wtg_num) + ' is read.')
    return wtg, wtg_normalInfo, wtg_failureInfo


# match tag
def match_tag(one_row, failureInfo, normalInfo):
    normal_tag = False
    failure_tag = False
    for j in range(0, len(failureInfo) - 1):
        if one_row['time'] >= failureInfo.iloc[j, 0] and one_row['time'] <= failureInfo.iloc[j, 1]:
                failure_tag = True

    if failure_tag:
        return one_row['tag'] + 2
    else:
        for k in range(0, len(normalInfo) - 1):
            if one_row['time'] >= normalInfo.iloc[k, 0] and one_row['time'] <= normalInfo.iloc[k, 1]:
                normal_tag = True
        if normal_tag:
            return one_row['tag'] + 1



wtg_list = [15, 21]
for wtg_num in wtg_list:
    wtg, wtg_normalInfo, wtg_failureInfo = read_data(wtg_num)
    wtg['tag'] = -1

    print('Then match tag, this may take a long time, be patient...')
    wtg['tag'] = wtg.apply(match_tag, axis=1, args=(wtg_failureInfo, wtg_normalInfo))
    wtg = wtg.loc[lambda df: df['tag'] > -1, :]

    # save to file, change the directory here
    print('New File is saving...')
    wtg.to_csv(r'F:/Temp/'+str(wtg_num) + '_tagged.csv', index=False, encoding='utf-8')

# then read the new generated data for further operation
