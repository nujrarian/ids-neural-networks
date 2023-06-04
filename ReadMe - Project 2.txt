             -----------------------------------------------------------------
            |       INTRUSION DETECTION SYSTEM - USING NEURAL NETWORKS        |
             ----------------------------------------------------------------

Author: Arjun Nair (AXN2607)
	Sai Tarun Sathyan (SS4005)
        


Folder Format:

project {root folder}/ -> data-cleanup.py

	            -> data-compilation.py

	            -> IDS-NeuralNetwork.py
           
                    -> data {folder}/ -> 02-14-2018.csv

		 	                  -> 02-15-2018.csv

			                  -> 02-16-2018.csv

			                  -> 02-22-2018.csv

			                  -> 02-23-2018.csv

			                  -> 03-01-2018.csv

			                  -> 03-02-2018.csv

			                  -> cleaned {folder}

			                  -> results {folder} -> models {folder}



The contents and structure of the Project 2 folder is provided above.
The steps to run the program is provided below.

Step 1: Execute the data-cleanup.py file. This file will process and clean the entire dataset.
	  The cleaned CSV files will be written into the 'project/data/cleaned' folder.

Step 2: Execute the data-compilation.py file. This file will compile and classify the 7 csv files
	  into 2 new cleaned CSV files in the same folder. One file will be used as the Signature Based 
        IDS dataset while the other file will be used as the Anomaly Based IDS dataset.

Step 3: Execute the IDS-NeuralNetwork.py file. This file will train the neural network on the two
	  previously created Anomaly and Signature CSV files. It will create two models into 
	  the 'project/data/results/model' folder and print their respective accuracies.