# Intrusion Detection System using Neural Network

Authors: Arjun Nair and Sai Tarun Sathyan <br>

Dataset: https://www.unb.ca/cic/datasets/ids-2018.html

## Objective
The goal was to create a program to clean a huge dataset and create a sequential neural network model that is capable of identifying signature-based and anomaly-based attacks from the dataset. Our neural network model consists of 3 dense layers with the first two layers having a “ReLU” activation function and the final output layer having a “Softmax” activation function.<br>

## Results
The program produces an accuracy of 95% for signature-based attack classification and an accuracy of 98% for anomaly-based attack classification. <br>

The Training and testing results for the signature-based model:<br>
<img width="452" alt="image" src="https://github.com/nujrarian/ids-neural-networks/assets/55311409/d69e2a4e-2f04-4b5e-8f4b-2fddbf889669">
<br>
The confusion matrix for true/false positives is as follows:<br>
<img width="503" alt="image" src="https://github.com/nujrarian/ids-neural-networks/assets/55311409/984533d6-b508-4c08-952f-4a0b5a2c1ebd">
<br>
The Training and testing results for the anomaly-based model: <br>
<img width="452" alt="image" src="https://github.com/nujrarian/ids-neural-networks/assets/55311409/cd520733-e644-4148-8492-6c5a814484de">
<br>
The confusion matrix for true/false positives is as follows: <br>
<img width="287" alt="image" src="https://github.com/nujrarian/ids-neural-networks/assets/55311409/7d8615a8-6b2a-4ac3-92f4-ae7be64f11dc">
<br>

## Steps to run the program
Folder Format: <br>

project/data-cleanup.py <br>
project/data-compilation.py <br>
project/IDS-NeuralNetwork.py <br>
project/data/02-14-2018.csv <br>
project/data/02-15-2018.csv <br>
project/data/02-16-2018.csv <br>
project/data/02-22-2018.csv <br>
project/data/02-23-2018.csv <br>
project/data/03-01-2018.csv <br>
project/data/03-02-2018.csv <br>
project/data/cleaned <br>
project/data/results/models <br>

The contents and structure of the Project folder is provided above.
The steps to run the program is provided below.

Step 1: Execute the data-cleanup.py file. This file will process and clean the entire dataset. The cleaned CSV files will be written into the 'project/data/cleaned' folder.

Step 2: Execute the data-compilation.py file. This file will compile and classify the 7 csv files into 2 new cleaned CSV files in the same folder. One file will be used as the Signature Based IDS dataset while the other file will be used as the Anomaly Based IDS dataset.

Step 3: Execute the IDS-NeuralNetwork.py file. This file will train the neural network on the two previously created Anomaly and Signature CSV files. It will create two models into the 'project/data/results/model' folder and print their respective accuracies.
  
