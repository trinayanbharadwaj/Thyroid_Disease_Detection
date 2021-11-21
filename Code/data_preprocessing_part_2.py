# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler 
import logging

logging.basicConfig(filename="Data_proprocessing _part_2.log", level=logging.INFO, format='%(asctime)s %(message)s',
					datefmt="%Y-%m-%d %H:%M:%S")

logging.info("Reading the dataset.........")
df =  pd.read_csv("D:\iNeuron Internship\code\data_processed_1.csv")
logging.info("Read the dataset successfully.")

# We saw that the null values were imputed by a "?". To keep things simple I re-converted the "?" into numpy null value.
df.replace({"?":np.nan}, inplace=True)
logging.info("Replaced all ? into Null values.")

# The columns "TBG" and "T3" has a lot of null values. Imputing these null values might reduce variance and might tease 
# the model into giving more importance to a particular case.
df.drop(columns=["TBG", "T3"], inplace=True)
logging.info("Droped column TBG and T3 for high missing values.")

# The column "sex" has a few null values and I thought imputing them with "unknown" makes more sense 
# than male "M" or "female" "F".
df.sex.fillna("unknown", inplace=True)

# Coverting the datatype of continous features into numeric type.
df.TSH = pd.to_numeric(df.TSH)
df.TT4 = pd.to_numeric(df.TT4)
df.T4U = pd.to_numeric(df.T4U)
df.FTI = pd.to_numeric(df.FTI)

# Removing outliers 
index_age = df[df["age"]>100].index
df.drop(index_age, inplace=True)
logging.info("Droped outliers from age feature.")

# removing TSH value higher than 15. That's quiet rare.
index_tsh = df[df["TSH"]>15].index
df.drop(index_tsh, inplace=True)
logging.info("Droped outliers from TSH feature.")

# Encoding the categorical features. 
df_dummy = pd.get_dummies(df)
logging.info("Encoded the categorical features.")

# Imputing null values using KNNImputer.
logging.info("Imputation of trainind and testing missing values with KNNImputer initiated......")
def Imputation(df):
    imputer = KNNImputer(n_neighbors=3)
    df_1 = imputer.fit_transform(df)
    df_2 = pd.DataFrame(df_1, columns=df.columns)
    return df_2
    

df_final = Imputation(df_dummy[:7000])
logging.info("Successfully imputed the missing values in train and test.")

# Splitting the data into train, test and validation to prevent data leakage.
validation_data = df_dummy[7000:]
x_train, x_test, y_train, y_test = train_test_split(df_final.drop(columns="outcome"), df_final["outcome"], test_size=0.2)
logging.info("Created training, testing and validation dataset.")

logging.info("Initiated imputation in validation dataset.")
valid_final = Imputation(validation_data)
logging.info("Successfully imputed in validation dataset.")

# Fixing the imbalanced data by creating duplicate records.
logging.info("Fixing imbalanced data initiated.....")
def balance_data(x,y):    
    ros = RandomOverSampler(random_state=42)
    x_sample, y_sample = ros.fit_resample(x, y)
    return x_sample, y_sample

x_train, y_train = balance_data(x_train, y_train)
logging.info("fixed imbalanced data in train dataset.")
x_test, y_test = balance_data(x_test, y_test)
logging.info("fixed imbalanced data in test dataset.")
x_valid, y_valid = balance_data(valid_final.drop(columns="outcome"), valid_final["outcome"])
logging.info("fixed imbalanced data in validation dataset.")

logging.info("saving the training data.....")
x_train.to_csv("x_train.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
logging.info("Successfully saved the training data.")

logging.info("saving the testing data.....")
x_test.to_csv("x_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
logging.info("Successfully saved the testing data.")

logging.info("saving the validation data......")
x_valid.to_csv("x_valid.csv", index=False)
y_valid.to_csv("y_valid.csv", index=False)
logging.info("Successfully saved the validation data.")