import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tcn.TemporalConvNet import *
from TCNDataGenerator import *

try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras


seq_len = 7
input_channels = 34

# get train and test data
train = pd.read_csv('data/train.csv', parse_dates=[0], dtype={'tag': 'int64'})
test = pd.read_csv('data/test.csv', parse_dates=[0], dtype={'tag': 'int64'})
X_cols = ['wind_speed', 'generator_speed', 'power', 'wind_direction',
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
train_X, train_Y = data_generator.slides_aligned_sample(20000)
test_X, test_Y = data_generator.get_test()

# divide train data to train and valid data
valid_X, valid_Y = train_X[35000:, :], train_Y[35000:]
train_X, train_Y = train_X[0: 35000, :], train_Y[0: 35000]

train_X = tf.constant(train_X, shape=(35000, 7, 34))
train_Y = tf.constant(train_Y, shape=(35000, 1, 1))
train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(30)

vaild_X = tf.constant(valid_X, shape=(5000, 7, 34))
valid_Y = tf.constant(valid_Y, shape=(5000, 1, 1))
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_X, valid_Y))
valid_dataset = valid_dataset.shuffle(buffer_size=1024).batch(10)

tcn_model = TemporalConvNet(input_channels=input_channels, layers_channels=[128, 64, 16, 4, 1], kernel_size=3)
tcn_model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01), loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

callbacks = [keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=1e-3,
    patience=100,
    mode='min',
    verbose=2
)]
tcn_model.fit(train_dataset, validation_data=valid_dataset, callbacks=callbacks, epochs=1000, verbose=2)

"""
test_x = tf.reshape(test_X, [len(test_X), seq_len, input_channels])
test_x_pred = tcn_model.predict(test_x)
pred_y = []

for i in test_x_pred:
    pred_y.append(i[-1])

inverse_pred_y = pred_y
inverse_test_y = test_Y
total_redeem_amt_pred = inverse_pred_y[:, 0]
total_purchase_amt_pred = inverse_pred_y[:, 1]
total_redeem_amt_value = inverse_test_y[:, 0]
total_purchase_amt_value = inverse_test_y[:, 1]
"""
"""
report_date = ['2014-08-22', '2014-08-23', '2014-08-24', '2014-08-25', '2014-08-26', '2014-08-27', '2014-08-28',
               '2014-08-29', '2014-08-30', '2014-08-31']
df = pd.DataFrame(
    data={'total_redeem_amt_pred': total_redeem_amt_pred, 'total_redeem_amt_value': total_redeem_amt_value,
          'total_purchase_amt_pred': total_purchase_amt_pred,
          'total_purchase_amt_value': total_purchase_amt_value}, index=report_date)

plt.figure(figsize=(18, 12))
plt.subplot(211)
plt.title('total_redeem_amt')
plt.plot(df['total_redeem_amt_pred'], label='total_redeem_amt_pred', color='blue')
plt.plot(df['total_redeem_amt_value'], label='total_redeem_amt_value', color='red')
plt.legend(loc='best')

plt.subplot(212)
plt.title('total_purchase_amt')
plt.plot(df['total_purchase_amt_pred'], label='total_purchase_amt_pred', color='blue')
plt.plot(df['total_purchase_amt_value'], label='total_purchase_amt_value', color='red')
plt.legend(loc='best')

plt.show()
"""