import pandas as pd
"""
This code is for tag matching,
dealing with both 15 and 21 wtg.
"""


# change the directory here to your own directory
# and also remember to change the diretory in the end of this code
# read data according to the number of wtg
def read_data(wtg_num):
    # the directory of data
    directory = r'E:/Docs/Jupyter/WTG_Ice_Prediction/Data/' + str(wtg_num) + r'/' + str(wtg_num)
    # read in data and parse date of column 0
    wtg = pd.read_csv(directory + r'_data.csv', parse_dates=[0])
    # read in tag files
    wtg_normalInfo = pd.read_csv(directory + r'_normalInfo.csv', parse_dates=[0,1])
    wtg_failureInfo = pd.read_csv(directory + r'_failureInfo.csv', parse_dates=[0,1])
    print('data of wtg ' + str(wtg_num) + ' is read.')
    return wtg, wtg_normalInfo, wtg_failureInfo


# match tag function for pa.apply() row by row
def match_tag(one_row, failureInfo, normalInfo):
    # set flag of normal and failure
    normal_tag = False
    failure_tag = False
    # a loop going through failureInfo
    for j in range(0, len(failureInfo) - 1):
        # if data time in failureInfo, set flag of failure_tag true
        if one_row['time'] >= failureInfo.iloc[j, 0] and one_row['time'] <= failureInfo.iloc[j, 1]:
                failure_tag = True
    # if failure_tag true, then set data tag 1
    if failure_tag:
        return one_row['tag'] + 2
    # if failure tag false, loop over normalInfo
    else:
        for k in range(0, len(normalInfo) - 1):
            # if data time in normalInfo, set flag of normal_tag true
            if one_row['time'] >= normalInfo.iloc[k, 0] and one_row['time'] <= normalInfo.iloc[k, 1]:
                normal_tag = True
        # if normal_tag is true, then set data tag 0
        if normal_tag:
            return one_row['tag'] + 1
    # if data time neither in failureInfo nor in normalInfo, then data tag remain -1


# num list of 2 wtg
wtg_list = [15, 21]
# do tag matching to each wtg
for wtg_num in wtg_list:
    # call read_data function to read in data and normalInfo and failureInfo
    wtg, wtg_normalInfo, wtg_failureInfo = read_data(wtg_num)
    # first set all tag in data -1
    wtg['tag'] = -1

    print('Then match tag, this may take a long time, be patient...')
    # call match_tag function applying to each row in data and update tag in data
    wtg['tag'] = wtg.apply(match_tag, axis=1, args=(wtg_failureInfo, wtg_normalInfo))
    # filter abnormal data whose time neither in normalInfo nor in failureInfo according to tag = -1
    wtg = wtg.loc[lambda df: df['tag'] > -1, :]

    # save to file, change the directory here
    print('New File is saving...')
    wtg.to_csv(r'F:/Temp/'+str(wtg_num) + '_tagged.csv', index=False, encoding='utf-8')

# then read the new generated data for further operation
