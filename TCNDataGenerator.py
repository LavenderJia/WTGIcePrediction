import pandas as pd
import numpy as np


class DataGenerator(object):
    """
    The interface of construct train, test, validation data
    We consider two parts in data generating:
    1. construct X_, Y_
    2. sample and division, to get train_X, train_Y, validation_X, validation_Y, test_X, test_Y
    Both of these two steps may have several implement, so we have interface here
    """
    def cons_series_fore_padding(self, padding_len: int, interval: int) -> None:
        """
        just padding once at the beginning of dataframe according to window_length
        :param padding_len: length of time window
        :param interval: not zero, better be smaller than padding_len
        :return: None
        """
        pass

    def slides_aligned_sample(self, num) -> (np.ndarray, np.array):
        """
        sample the input series to make the two categories have nearly the same size
        we may later add the regularization to loss function, then we do not need to sample
        so this function may later be abolished...
        :param num:
        :return: sampled data
        """
        pass

    def get_test(self) -> (np.ndarray, np.array):
        """

        :return:
        """

    def get_train(self) -> (np.ndarray, np.array):
        """
        get train_data
        :return:
        """


class DataGeneratorImpl(DataGenerator):
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, x_cols, y_cols):
        self.train_X = train.loc[:, x_cols].values
        self.train_Y = train.loc[:, y_cols].values.reshape(1, -1)[0]
        self.test_X = test.loc[:, x_cols].values
        self.test_Y = test.loc[:, y_cols].values.reshape(1, -1)[0]

    def get_test(self):
        return self.test_X, self.test_Y

    def get_train(self):
        return self.train_X, self.train_Y

    def cons_series_fore_padding(self, padding_len: int, interval: int):
        padding_zeros = np.zeros((padding_len, self.train_X.shape[1]))
        self.train_X = np.vstack((padding_zeros, self.train_X))
        self.test_X = np.vstack((padding_zeros, self.test_X))
        # fore-sample data to construct time series
        self.train_X = self.train_X[np.fromfunction(
            lambda i, j: i + padding_len - j * interval, (len(self.train_X) - padding_len, padding_len // interval + 1),
            dtype=int), :]
        self.test_X = self.test_X[np.fromfunction(
            lambda i, j: i + padding_len - j * interval, (len(self.test_X) - padding_len, padding_len // interval + 1),
            dtype=int), :]

    def slides_aligned_sample(self, num):
        sample_idx = np.hstack((
             np.random.choice(np.argwhere(self.train_Y == 0).reshape(1, -1)[0], num, replace=False),
             np.random.choice(np.argwhere(self.train_Y == 1).reshape(1, -1)[0], num, replace=False)))
        sample_idx.sort()
        return self.train_X[sample_idx, :], self.train_Y[sample_idx]


if __name__ == "__main__":
    train = pd.read_csv('data/train.csv', parse_dates=[0], dtype={'tag': 'int64'})
    test = pd.read_csv('data/test.csv', parse_dates=[0], dtype={'tag': 'int64'})

    X_cols = ['time', 'wind_speed', 'generator_speed', 'power', 'wind_direction',
                'wind_direction_mean', 'yaw_position', 'yaw_speed', 'pitch1_angle',
                'pitch2_angle', 'pitch3_angle', 'pitch1_speed', 'pitch2_speed',
                'pitch3_speed', 'pitch1_moto_tmp', 'pitch2_moto_tmp', 'pitch3_moto_tmp',
                'acc_x', 'acc_y', 'environment_tmp', 'int_tmp', 'pitch1_ng5_tmp',
                'pitch2_ng5_tmp', 'pitch3_ng5_tmp', 'pitch1_ng5_DC', 'pitch2_ng5_DC',
                'pitch3_ng5_DC', 'group', 'tmp_diff', 'torque', 'cp', 'ct',
                'lambda', 'pitch_angle_mean', 'pitch_angle_sd']
    Y_cols = ['tag']

    data_generator: DataGenerator = DataGeneratorImpl(train, test, X_cols, Y_cols)
    data_generator.cons_series_fore_padding(30, 5)
    train_X, train_Y = data_generator.slides_aligned_sample(100)
    test_X, test_Y = data_generator.get_test()
    print(train_X[0:10])
    print(train_Y[0:10])
    print(test_X[0:10])
    print(test_Y[0:10])















