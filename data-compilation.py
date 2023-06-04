"""
             -----------------------------------------------------------------
            |       INTRUSION DETECTION SYSTEM - USING NEURAL NETWORKS        |
             ----------------------------------------------------------------
Author: Sai Tarun Sathyan (SS4005)
        Arjun Nair (AXN2607)


Brief : This program combines the seperate  cleaned files
        It creates two CSVs , one with proper labelling (Signature Based)
        The 2nd one labels attacks as 1 and normal packets as 0 (Anomaly Based)
"""

import csv
import os
import sys
from sklearn.utils import shuffle
import numpy as np
import pandas as pd




# <<<<<<<<<<<<<   MAIN   >>>>>>>>>>>>>

dataPath = 'data/cleaned'
fileNames = ['02-14-2018-Cleaned.csv',
             '02-15-2018-Cleaned.csv',
             '02-16-2018-Cleaned.csv',
             '02-22-2018-Cleaned.csv',
             '02-23-2018-Cleaned.csv',
             '03-01-2018-Cleaned.csv',
             '03-02-2018-Cleaned.csv']


# <<< Appending every file into one >>>
df = pd.read_csv(os.path.join(dataPath, fileNames[0]))
print(df.shape)
for name in fileNames[1:]:
    fname = os.path.join(dataPath, name)
    print('appending:', fname)
    df1 = pd.read_csv(fname)
    df = df.append(df1, ignore_index=True)


# <<< Shuffling the combined CSV file >>>
df = shuffle(df)
print(df.shape)
print('Dataset-Signature CSV file...')
# <<< Creating Signature Based CSV file >>>
outFile = os.path.join(dataPath, 'CIC-Dataset-Signature-Cleaned')
df.to_csv(outFile + '.csv', index=False)
df.to_pickle(outFile + '.pickle')


print('creating Dataset-Anomaly CSV file...')
# <<< Creating Anomaly Based CSV file >>>
df['Label'] = df['Label'].map(
    {'Benign': 0,
     'FTP-BruteForce': 1,
     'SSH-Bruteforce': 1,
     'DoS attacks-GoldenEye': 1,
     'DoS attacks-Slowloris': 1,
     'DoS attacks-SlowHTTPTest': 1,
     'DoS attacks-Hulk': 1,
     'Brute Force -Web': 1,
     'Brute Force -XSS': 1,
     'SQL Injection': 1,
     'Infiltration': 1,
     'Bot': 1})
print(df['Label'][1:20])
outFile = os.path.join(dataPath, 'CIC-Dataset-Anomaly-Cleaned')
df.to_csv(outFile + '.csv', index=False)
df.to_pickle(outFile + '.pickle')

print('Completed Data Compilation and Classification')
