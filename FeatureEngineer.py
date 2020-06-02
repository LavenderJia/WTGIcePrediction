import pandas as pd
from datetime import datetime,timedelta
"""
1. Deal with noise in the data: moving average and delete outliers. specifically, we design a interval moving 
average method, in which we consider time intervals of two records, if the interval of the two successive 
records is longer than 10 minutes, we skip such record without averaging it. And noting that we only delete 
outliers in 15 and only in normal data. 
2. Simple construction of some features  
3. do data standardize to new constructed features
"""


# read data
# change the directory here to your own directory
# and also remember to change the diretory in the end of this code
# the function for reading tagged data according to the number of wtg
def read_data(wtg_num):
    # parse dtype of column 0 to date and set tag column to int64
    wtg = pd.read_csv(r'F:/Temp/Tag/' + str(wtg_num) + r'_tagged.csv', parse_dates=[0], dtype={'tag': 'int64'})
    print('Data of wtg ' + str(wtg_num) + ' is read.')
    return wtg


# do diff to time to get time interval
def get_record_interval(df):
    df.loc[:, 'rec_time_interval'] = df.time.diff()  # time interval of two records
    return df


# interval moving average
def interval_moving_average(df, calculate_cols):
    #  get index where data interval is larger then 30s
    sep_index = df.loc[lambda df: df['rec_time_interval'] > timedelta(seconds=30), :].index
    # set first start index zero
    start_index = 0
    # loop over intervals in sep index and to moving average to every data part
    for index in sep_index:
        # set end_index as the current index
        end_index = index
        # do moving average
        df.loc[start_index:end_index, calculate_cols] = df.loc[start_index:end_index, calculate_cols].rolling(3, win_type=None, center=True, min_periods=1).mean()
        # set next start index as end index to start next step of loop
        start_index = end_index
    return df


# for extreme outliers: Q3 + 3 * (Q3 - Q1); Q1 - 3 * (Q3 - Q1)
# just delete outliers in normal data
def outlier_detector_for_normal(df, cols):
    # the list for collection of index of outlier index
    index = []
    # for columns to detect outliers
    for col in cols:
        # calculate upper bound of the column
        upper = df.loc[lambda df: df['tag'] == 0, col].quantile(0.75) + 3 * (df.loc[lambda df: df['tag'] == 0, col].quantile(0.75) -
                                 df.loc[lambda df: df['tag'] == 0, col].quantile(0.25))
        # calculate lower bound of the column
        lower = df.loc[lambda df: df['tag'] == 0, col].quantile(0.25) - 3 * (df.loc[lambda df: df['tag'] == 0, col].quantile(0.75) -
                                 df.loc[lambda df: df['tag'] == 0, col].quantile(0.25))
        # add outlier index to index list
        index.extend(list(df.loc[lambda df: df['tag'] == 0 , :].loc[lambda df: df[col] > upper, : ].index))
        index.extend(list(df.loc[lambda df: df['tag'] == 0 , :].loc[lambda df: df[col] < lower, : ].index))
        # drop duplicates in index list
        index = list(set(index))
    return index


# construct new features
def feature_constructor(df):
    # temperature difference of inner and outer environment
    df.loc[:, 'tmp_diff'] = df.loc[:, 'int_tmp'] - df.loc[:, 'environment_tmp']
    df.loc[:, 'torque'] = df.loc[:, 'power'] / df.loc[:, 'generator_speed']
    # power coefficient
    df.loc[:, 'cp'] = df.loc[:, 'power'] / (df.loc[:,'wind_speed'] ** 3)
    #  thrust coefficient
    df.loc[:, 'ct'] = df.loc[:, 'torque'] / (df.loc[:,'wind_speed'] ** 2)
    # rate of wind_speed to power
    df.loc[:, 'r_windspeed_to_power'] = ((df.loc[:,'wind_speed'] + 5) / (df.loc[:, 'power'] + 5))**2 - 1
    # rate of wind_speed to generator_speed
    df.loc[:, 'r_windspeed_to_generator_speed'] = ((df.loc[:, 'wind_speed'] + 5) / (df.loc[:, 'generator_speed'] + 5))**2 - 1
    # rate of wind_speed to power*generator
    df.loc[:, 'r_square'] = ((df.loc[:, 'wind_speed'] + 5)**2 / ((df.loc[:, 'generator_speed'] + 5)*(df.loc[:, 'power'] + 5)))**2 - 1

    # mean of pitches' angle
    df.loc[:, 'pitch_angle_mean'] = df.loc[:, ['pitch1_angle','pitch2_angle','pitch3_angle']].mean(axis=1)
    # standard deviation of pitches' angle
    df.loc[:, 'pitch_angle_sd'] = df.loc[:, ['pitch1_angle','pitch2_angle','pitch3_angle']].std(axis=1)
    # mean of pitches' speed
    df.loc[:, 'pitch_speed_mean'] = df.loc[:, ['pitch1_speed','pitch2_speed','pitch3_speed']].mean(axis=1)
    # standard deviation of pitches' speed
    df.loc[:, 'pitch_speed_sd'] = df.loc[:, ['pitch1_speed','pitch2_speed','pitch3_speed']].std(axis=1)
    # mean of moto tmp
    df.loc[:, 'moto_tmp_mean'] = df.loc[:, ['pitch1_moto_tmp','pitch2_moto_tmp','pitch3_moto_tmp']].mean(axis=1)
    # standard deviation of pitches' moto tmp
    df.loc[:, 'moto_tmp_sd'] = df.loc[:, ['pitch1_moto_tmp','pitch2_moto_tmp','pitch3_moto_tmp']].std(axis=1)

    # diff of mean of pitches' angle
    df.loc[:, 'diff_pitch_angle'] = df.loc[:, 'pitch_angle_mean'].diff()
    # diff of mean of moto tmp
    df.loc[:, 'diff_moto_tmp'] = df.loc[:, 'moto_tmp_mean'].diff()
    # diff of ng5_tmp
    df.loc[:, 'diff_pitch1_ng5_tmp'] = df.loc[:, 'pitch1_ng5_tmp'].diff()
    df.loc[:, 'diff_pitch2_ng5_tmp'] = df.loc[:, 'pitch2_ng5_tmp'].diff()
    df.loc[:, 'diff_pitch3_ng5_tmp'] = df.loc[:, 'pitch3_ng5_tmp'].diff()
    # df.dropna(inplace=True)
    return df


# standardize some new features: one-zero
def one_zero_standardize(df, cols):
    df.loc[:, cols] = (df.loc[:, cols]-df.loc[:, cols].mean())/df.loc[:, cols].std()
    return df


# standardize some new feature: max_min
def max_min_standardize(df, cols):
    df.loc[:,cols] = (df.loc[:,cols] - df.loc[:, cols].min())/(df.loc[:,cols].max() - df.loc[:,cols].min())
    return df


# transfer rec_time_interval to seconds, used in df.apply()
def get_interval_seconds(row):
    row['rec_time_interval'] = row['rec_time_interval'].total_seconds()
    return row['rec_time_interval']


# organize features
def keep_features(df, cols):
    return df.loc[:, cols]


# add lag of variables by interval of 40, 80, 160, 240, 400, 560
def add_lag(df, cols):
    for col in cols:
        df.loc[:, col + '_lag40'] = df.loc[:, col].diff(periods=40)
        df.loc[:, col + '_lag80'] = df.loc[:, col].diff(periods=80)
        df.loc[:, col + '_lag160'] = df.loc[:, col].diff(periods=160)
        df.loc[:, col + '_lag240'] = df.loc[:, col].diff(periods=240)
        df.loc[:, col + '_lag400'] = df.loc[:, col].diff(periods=400)
        df.loc[:, col + '_lag560'] = df.loc[:, col].diff(periods=560)
    return df


# the process function calling the above functions
def run(max_min=False, lag=False):
    # the number list of wtg
    wtg_list = [15, 21]
    # do process to each wtg
    for wtg_num in wtg_list:
        # set res file name
        res_file_name = str(wtg_num) + '_FE'
        # read data of wtg
        wtg = read_data(wtg_num)
        # get record interval
        wtg = get_record_interval(wtg)
        # do moving average
        wtg = interval_moving_average(wtg, wtg.columns[1:26])
        print('moving average is finished.')
        # columns for outlier detecting
        outlier_detector_columns = list(wtg.columns)[1: -3]
        # add time series info, lagged columns
        lag_columns = []
        # if lag param is True
        if lag:
            # add lag
            wtg = add_lag(wtg, ['wind_speed', 'environment_tmp'])
            print('lags are added.')
            # change res file name
            res_file_name += '_TSInfo'
            lag_columns = list(wtg.columns)[-12:]
        # if wtg num is 15, delete outliers
        if wtg_num == 21:
            normal_outlier_index = outlier_detector_for_normal(wtg, outlier_detector_columns)
            wtg = wtg.iloc[list(set(wtg.index) - set(normal_outlier_index)), :]
            wtg.reset_index(inplace=True)
            del wtg['index']
            wtg = get_record_interval(wtg)  # recalculate record interval
            print('Outliers of normal is deleted.')
        # construct new features
        wtg = feature_constructor(wtg)
        print('New features is constructed.')
        # transfer rec_time_interval to seconds
        wtg['rec_time_interval'] = wtg.apply(get_interval_seconds, axis=1)
        print('Record time interval is transformed to seconds.')
        # standardize some new features
        # if max_min param is true, use max_min_standardize
        if max_min:
            wtg = max_min_standardize(wtg, ['torque', 'cp', 'ct', 'rec_time_interval'])
            res_file_name += '_MMS'
        # if max_min param is false, use one_zero_standardize
        else:
            wtg = one_zero_standardize(wtg, ['torque', 'cp', 'ct', 'rec_time_interval'])
            res_file_name += '_OZS'
        print('Standardize of torque, cp, ct, rec_time_interval is done.')
        # arrange data columns, drop some old features
        keeped_features = ['time', 'rec_time_interval',
                           'wind_speed', 'wind_direction', 'wind_direction_mean',
                           'generator_speed', 'power',
                           # 'yaw_position',
                           'yaw_speed',
                           'acc_x',
                           # 'acc_y',
                           'environment_tmp',
                           # 'int_tmp',
                           'pitch1_ng5_tmp', 'pitch2_ng5_tmp', 'pitch3_ng5_tmp',
                           'pitch1_ng5_DC', 'pitch2_ng5_DC',
                           # 'pitch3_ng5_DC',
                           'tmp_diff', 'torque', 'cp', 'ct',
                           'r_windspeed_to_power', 'r_windspeed_to_generator_speed', 'r_square',
                           'pitch_angle_mean', 'pitch_angle_sd',
                           'pitch_speed_mean', 'pitch_speed_sd',
                           'moto_tmp_mean', 'moto_tmp_sd',
                           #'diff_pitch_angle',
                           'diff_moto_tmp',
                           'diff_pitch1_ng5_tmp', 'diff_pitch2_ng5_tmp', 'diff_pitch3_ng5_tmp'
                           ]
        # reset columns
        keeped_features.extend(lag_columns)
        keeped_features.extend(['tag'])
        wtg = keep_features(wtg, keeped_features)
        print('Columns is reset.')
        # save to file
        # you can later read such file for further analysis
        print('New File is saving...')
        wtg.to_csv(r'F:/Temp/FE Data V2/'+ res_file_name + '.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    # form different forms of data
    print('Do feature engineering with one zero standardize and without time series info.')
    run()
    print('Do feature engineering with max min standardize and without time series info.')
    run(max_min=True)
    print('Do feature engineering with one zero standardize and with time series info.')
    run(lag=True)
    print('Do feature engineering with max min standardize and with time series info.')
    run(max_min=True, lag=True)
