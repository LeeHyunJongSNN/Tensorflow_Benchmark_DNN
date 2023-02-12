import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.optimizers.optimizer_v2 import adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

max_len = 100
dropout_ratio = 0.2
hidden_units_1 = 10
hidden_units_2 = 5
hidden_units_3 = 3

test_valid_ratio = 0.3
test_ratio = 0.5

train_data = []
test_data = []

wave_data = []
wave_label = []

fname = " "
for fname in ["/home/leehyunjong/Dataset_2.4GHz/1kHz_100/amplitude/"
              "(sine+sawtooth)_1kHz_100_amplitude_18dB_60000.txt"]:

    print(fname)
    f = open(fname, "r", encoding='utf-8-sig')
    linedata = []

    for line in f:
        if line[0] == "#":
            continue

        linedata = [float(x) for x in line.split()]
        if len(linedata) == 0:
            continue

        raw = [x for x in linedata[0:len(linedata) - 1]]
        # dcr = np.round(np.absolute(detrend(raw - np.mean(raw))), 2)
        # scaled = scale * np.round(dcr / np.max(dcr), 2)

        cl = linedata[-1].real

        wave_label.append(cl)
        wave_data.append(raw)

    f.close()

train_data, test_valid_data, train_label, test_valid_label = train_test_split(
    wave_data, wave_label, test_size=test_valid_ratio)
valid_data, test_data, valid_label, test_label = train_test_split(
    test_valid_data, test_valid_label, test_size=test_ratio)

train_data = np.array(train_data).reshape(-1, max_len, 1)
valid_data = np.array(valid_data).reshape(-1, max_len, 1)
test_data = np.array(test_data).reshape(-1, max_len, 1)
train_label = np.array(train_label).reshape((-1, 1))
valid_label = np.array(valid_label).reshape((-1, 1))
test_label = np.array(test_label).reshape((-1, 1))

print('train size :', train_data.shape)
print('valid size :', valid_data.shape)
print('test size :', test_data.shape)

adam = adam.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

model = Sequential()
model.add(Dense(hidden_units_1, input_shape=(max_len, 1), activation='relu'))
model.add(Dense(hidden_units_2, activation='relu'))
model.add(Dense(hidden_units_3, activation='relu'))
model.add(Dropout(dropout_ratio))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])
history = model.fit(train_data, train_label, validation_data=(valid_data, valid_label), epochs=100,
                    batch_size=64, use_multiprocessing=True, callbacks=[es, mc])

# history = model.fit(train_data, train_label, validation_data=(valid_data, valid_label), epochs=400,
#                     batch_size=64, use_multiprocessing=True)

print("\n test accuracy: %.4f" % (model.evaluate(test_data, test_label, batch_size=64)[1]))

prediction = model.predict(test_data)
bin_prediction = tf.round(prediction)
print(classification_report(test_label, bin_prediction))
cm = confusion_matrix(test_label, bin_prediction)
print(cm)
print("Probability of Detection: %.4f" % (cm[0][0] / (cm[0][0] + cm[1][0])))
print("False Negative Probability: %.4f" % (cm[1][0] / (cm[0][0] + cm[1][0])))
print("False Positive Probability: %.4f" % (cm[0][1] / (cm[0][1] + cm[1][1])))
