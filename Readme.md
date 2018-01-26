# Author classification on the Enron email database

## -Python for text analysis 

## Script files
In the src folder:

1. Enron.ipynb -- as the main script
2. extract\_email\_info.py -- for extract information from emails
3. features.py --  for extract features for classification

## In The Enron.ipynb
### Input:

Enron email database from Kaggle website <https://www.kaggle.com/wcukierski/enron-email-dataset>

### Content:
1. Obtain the data
2. Extract information from dataset;
3. Choose reasonable data size for processing;
4. Using text analysis methods to extract features for later classification.
5. Visualize the statistics of data
6. Using machine learning method for an automatic classification task
7. Store the predictions

### Output files as results:
In the data folder:

1. Aggregate\_per\_author.csv
2. Statistics\_for\_single\_training\_example.csv
3. predictions\_test\_data.csv

### Other output files:
In the root folder( not included in git):

1. email\_csv.csv
--save email information after Message analysis step

2. emails\_df\_1000\_12\_tmp1.pkl

	emails\_df\_1000\_12\_tmp2.pkl
   	
   	emails\_df\_1000\_12\_tmp3.pkl
   	
   	emails\_df\_1000\_12\_tmp4.pkl

-- save doc structures after applying nlp

