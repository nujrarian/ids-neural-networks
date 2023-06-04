"""
             -----------------------------------------------------------------
            |       INTRUSION DETECTION SYSTEM - USING NEURAL NETWORKS        |
             ----------------------------------------------------------------
Author: Sai Tarun Sathyan (SS4005)
        Arjun Nair (AXN2607)


Brief : This program designs our neural network model
        Trains the model on our previously cleaned and classified dataset
        Produces test results
        Saves the trained model
"""

import csv
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sn
import operator

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical, normalize
from sklearn.utils import shuffle
from keras.callbacks import TensorBoard
from timeit import default_timer as timer
import time

dataPath = 'data/cleaned'
resultPath = 'data/results'
if not os.path.exists(resultPath):
    print('result path {} created.'.format(resultPath))
    os.mkdir(resultPath)

dep_var = 'Label'
model_name = "init"

cat_names = ['Dst Port', 'Protocol']
cont_names = ['Timestamp', 'Flow Duration', 'Tot Fwd Pkts',
              'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
              'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
              'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
              'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
              'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
              'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
              'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
              'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
              'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
              'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
              'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
              'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
              'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
              'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
              'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
              'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
              'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
              'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
              'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
              'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']


def loadData(fileName):
    dataFile = os.path.join(dataPath, fileName)
    pickleDump = '{}.pickle'.format(dataFile)
    if os.path.exists(pickleDump):
        df = pd.read_pickle(pickleDump)
    else:
        df = pd.read_csv(dataFile)
        df = df.dropna()
        df = shuffle(df)
        df.to_pickle(pickleDump)
    return df


def baseline_model(inputDim=-1, out_shape=(-1,)):
    global model_name
    model = Sequential()
    if inputDim > 0 and out_shape[1] > 0:
        model.add(Dense(79, activation='relu', input_shape=(inputDim,)))
        print(f"out_shape[1]:{out_shape[1]}")
        model.add(Dense(128, activation='relu'))

        model.add(Dense(out_shape[1], activation='softmax'))  # This is the output layer

        if out_shape[1] > 2:
            print('Categorical Cross-Entropy Loss Function')
            model_name += "_categorical"
            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        else:
            model_name += "_binary"
            print('Binary Cross-Entropy Loss Function')
            model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
    return model


def draw_confusion_matrix(model, X, y):
    XPrediction = model.predict(X, batch_size=32, verbose=1)
    Xpredicted = np.argmax(XPrediction, axis=1)
    confusionMatrix = confusion_matrix(np.argmax(y, axis=1), Xpredicted)
    matrix = pd.DataFrame(confusionMatrix, range(3), range(3))

    plt.figure(figsize = (3,3))
    sn.set(font_scale=1)
    ax = sn.heatmap(matrix, annot=True, annot_kws={"size":10})
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    FP = confusionMatrix.sum(axis=0) - np.diag(confusionMatrix)
    FN = confusionMatrix.sum(axis=1) - np.diag(confusionMatrix)
    TP = np.diag(confusionMatrix)
    TN = confusionMatrix.sum() - (FP + FN + TP)

    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    ACC = (TP+TN)/(TP+FP+FN+TN)

    print("Model Accuracy Details:-")
    print('True positive rate:', TPR)
    print('True negative rate:', TNR)
    print('False positive rate:', FPR)
    print('False negative rate:', FNR)


def experiment(dataFile, optimizer='adam', epochs=10, batch_size=10):
    time_gen = int(time.time())
    global model_name
    model_name = f"{dataFile}_{time_gen}"
    # $ tensorboard --logdir=logs/
    tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))

    seed = 7
    np.random.seed(seed)
    cvscores = []
    print('optimizer: {} epochs: {} batch_size: {}'.format(optimizer, epochs, batch_size))
    data = loadData(dataFile)
    data_y = data.pop('Label')

    # <<< Transforming named labels into numerical values >>>
    encoder = LabelEncoder()
    encoder.fit(data_y)
    data_y = encoder.transform(data_y)
    dummy_y = to_categorical(data_y)
    data_x = normalize(data.values)
    num = 0
    inputDim = len(data_x[0])
    print('inputdim = ', inputDim)

    # <<< Separating Training and Testing Data >>>
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
    start = timer()
    for train_index, test_index in sss.split(X=np.zeros(data_x.shape[0]), y=dummy_y):
        X_train, X_test = data_x[train_index], data_x[test_index]
        y_train, y_test = dummy_y[train_index], dummy_y[test_index]

        # <<< Creating The Model >>>
        model = baseline_model(inputDim, y_train.shape)

        # <<< Training the Model >>>
        print("Training " + dataFile + " on split " + str(num))
        model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[tensorboard],
                  validation_data=(X_test, y_test))

        # <<< Saving The Model >>>
        model.save(f"{resultPath}/models/{model_name}.model")

        num += 1

    elapsed = timer() - start

    scores = model.evaluate(X_test, y_test, verbose=1)
    print(model.metrics_names)
    acc, loss = scores[1] * 100, scores[0] * 100
    print('Baseline: accuracy: {:.2f}%: loss: {:.2f}'.format(acc, loss))

    resultFile = os.path.join(resultPath, dataFile)
    with open('{}.result'.format(resultFile), 'a') as fout:
        fout.write('{} results...'.format(model_name))
        fout.write('\taccuracy: {:.2f} loss: {:.2f}'.format(acc, loss))
        fout.write('\telapsed time: {:.2f} sec\n'.format(elapsed))


# <<<<<<<<<<<<<   MAIN   >>>>>>>>>>>>>
experiment("CIC-Dataset-Anomaly-Cleaned.csv")
experiment("CIC-Dataset-Signature-Cleaned.csv")