# Author classification on the Enron email database


This project extracts NLP features from Enron email database, and predicts the authors of emails. 
 

## File structure
- src folder
	- Enron.ipynb: main jupyter notebook script
	- extract\_email\_info.py: functions to get structure information from emails
	- features.py: functions to extract features for classification
- doc folder
	- enron\_email\_jing.key: presentation slides
	- Final\_Assignment_2017-2018.pdf: project manual
- result folder: required csv files
	- Aggregate\_per\_author.csv
	- predictions\_test\_data.csv
	- Statistics\_for\_single\_training\_example.csv


## Input
Download enron email database from Kaggle website <https://www.kaggle.com/wcukierski/enron-email-dataset>

## Pipelin
1. Obtain the data
2. Extract information from dataset
3. Choose reasonable data size for processing;
4. Using text analysis methods to extract features for later classification.
5. Visualize the statistics of data
6. Using machine learning method for an automatic classification task
7. Store the predictions

## Output files
- Aggregate\_per\_author.csv  :aggregated statistics on all features based on each user
- Statistics\_for\_single\_training\_example.csv  : statistics on all features for all samples
- predictions\_test\_data.csv   :the predictions on the test data



