# Importing the necessary libraries
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
import logging

logging.basicConfig(filename="Feature_selection.log", level=logging.INFO, format='%(asctime)s %(message)s',
					datefmt="%Y-%m-%d %H:%M:%S")

logging.info("Reading training dataset.........")
x_train = pd.read_csv("D:/iNeuron Internship/code/x_train.csv")
y_train = pd.read_csv("D:\iNeuron Internship\code\y_train.csv")
logging.info("Read the training data successfully.")

logging.info("Reading testing dataset........")
x_test = pd.read_csv("D:/iNeuron Internship/code/x_test.csv")
y_test = pd.read_csv("D:\iNeuron Internship\code\y_test.csv")
logging.info("Read the testing data successfully.")

# Applying selectKbest() to reduce the number of features.

def feature_selection(x,y):
   
    obj = SelectKBest(chi2, k=4)
    obj.fit_transform(x,y)
    filter = obj.get_support()
    feature = x.columns
    final_f = feature[filter]
    print(final_f)
    
    return final_f

logging.info("4 best features are............")
features = feature_selection(x_train, y_train)
logging.info(features)
