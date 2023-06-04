"""
             -----------------------------------------------------------------
            |       INTRUSION DETECTION SYSTEM - USING NEURAL NETWORKS        |
             ----------------------------------------------------------------
Author: Sai Tarun Sathyan (SS4005)
        Arjun Nair (AXN2607)


Brief : This program is used to clean a CSV file
        It removes : "Nan", "Inf", empty rows
        Saves the cleaned CSVs into a separate folder
"""



import os
import csv
import sys
import datetime
from dateutil import parser
from collections import defaultdict


def cleanData(inFile, outFile):
    count = 1
    stats = {}
    dropStats = defaultdict(int)
    print('cleaning {}'.format(inFile))
    with open(inFile, 'r') as csvfile:
        data = csvfile.readlines()
        totalRows = len(data)
        print('total rows read = {}'.format(totalRows))
        header = data[0]
        for line in data[1:]:
            line = line.strip()
            cols = line.split(',')
            key = cols[-1]
            if line.startswith('D') or line.find('Infinity') >= 0 or line.find('infinity') >= 0:
                dropStats[key] += 1
                continue

            dt = parser.parse(cols[2])  # '1/3/18 8:17'
            epochs = (dt - datetime.datetime(1970, 1, 1)).total_seconds()
            cols[2] = str(epochs)
            line = ','.join(cols)
            # clean_data.append(line)
            count += 1

            if key in stats:
                stats[key].append(line)
            else:
                stats[key] = [line]


    with open(outFile+".csv", 'w') as csvoutfile:
        csvoutfile.write(header)
        with open(outFile + ".stats", 'w') as fout:
            fout.write('Total Clean Rows = {}; Dropped Rows = {}\n'.format(
                count, totalRows - count))
            for key in stats:
                fout.write('{} = {}\n'.format(key, len(stats[key])))
                line = '\n'.join(stats[key])
                csvoutfile.write('{}\n'.format(line))
                with open('{}-{}.csv'.format(outFile, key), 'w') as labelOut:
                    labelOut.write(header)
                    labelOut.write(line)
            for key in dropStats:
                fout.write('Dropped {} = {}\n'.format(key, dropStats[key]))

    print('Selected rows: {}\nDropped rows: {}'.format(
        count, totalRows - count))


def cleanAllData():
    inputDataPath = '/data/'
    outputDataPath = '/data/cleaned/'
    if (not os.path.exists(outputDataPath)):
        os.mkdir(outputDataPath)

    files = os.listdir(inputDataPath)
    for file in files:
        if file.startswith('.'):
            continue
        if os.path.isdir(file):
            continue
        outFile = os.path.join(outputDataPath, file)
        inputFile = os.path.join(inputDataPath, file)
        cleanData(inputFile, outFile)



# <<<<<<<<<<<<<   MAIN   >>>>>>>>>>>>>

cleanData("data/02-14-2018.csv", "data/cleaned/02-14-2018-Cleaned")
cleanData("data/02-15-2018.csv", "data/cleaned/02-15-2018-Cleaned")
cleanData("data/02-16-2018.csv", "data/cleaned/02-16-2018-Cleaned")
cleanData("data/02-22-2018.csv", "data/cleaned/02-22-2018-Cleaned")
cleanData("data/02-23-2018.csv", "data/cleaned/02-23-2018-Cleaned")
cleanData("data/03-01-2018.csv", "data/cleaned/03-01-2018-Cleaned")
cleanData("data/03-02-2018.csv", "data/cleaned/03-02-2018-Cleaned")
print("Completed Data Cleanup")