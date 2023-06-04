# Intrusion Detection System using Neural Network

Authors: Arjun Nair (AXN2607) and Sai Tarun Sathyan (SS4005) <br>

Dataset: https://www.unb.ca/cic/datasets/ids-2018.html

## Objective
Thegoal was to create a program to clean a huge dataset and create a sequential neural network model that is capable of identifying signature-based and anomaly-based attacks from the dataset. Our neural network model consists of 3 dense layers with the first two layers having a “ReLU” activation function and the final output layer having a “Softmax” activation function. Major benefits of ReLUs 
are sparsity and a reduced likelihood of vanishing gradient. <br>
ReLu and its derivative is faster to compute than the sigmoid function. The Softmax activation function is able to handle multiple classes. It normalizes the outputs for each class between 0 
and 1 and divides by their sum. Hence forming a probability distribution. Therefore, giving a clear probability of input belonging to any particular class. The neural network model uses Adam optimizer simply because it’s the most frequently used optimizer and has the fastest computation time from our testing.

## Results
The Training and testing results for the signature-based model:

The confusion matrix for true/false positives is as follows:

The Training and testing results for the anomaly-based model:

The confusion matrix for true/false positives is as follows:

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

Step 1: Execute the data-cleanup.py file. This file will process and clean the entire dataset.
	  The cleaned CSV files will be written into the 'project/data/cleaned' folder.

Step 2: Execute the data-compilation.py file. This file will compile and classify the 7 csv files
	  into 2 new cleaned CSV files in the same folder. One file will be used as the Signature Based 
        IDS dataset while the other file will be used as the Anomaly Based IDS dataset.

Step 3: Execute the IDS-NeuralNetwork.py file. This file will train the neural network on the two
	  previously created Anomaly and Signature CSV files. It will create two models into 
	  the 'project/data/results/model' folder and print their respective accuracies.
    
    
