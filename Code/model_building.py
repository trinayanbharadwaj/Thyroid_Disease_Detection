# Importing the necessary libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
import pickle
import logging

logging.basicConfig(filename="Model_creation.log", level=logging.INFO, format='%(asctime)s %(message)s',
					datefmt="%Y-%m-%d %H:%M:%S")

logging.info("Reading training dataset.........")
x_train =  pd.read_csv("D:/iNeuron Internship/code/x_train.csv")
y_train =  pd.read_csv("D:/iNeuron Internship/code/y_train.csv")
logging.info("Read the training data successfully.")

logging.info("Reading testing dataset........")
x_test =  pd.read_csv("D:/iNeuron Internship/code/x_test.csv")
y_test =  pd.read_csv("D:/iNeuron Internship/code/y_test.csv")
logging.info("Read the testing data successfully.")

logging.info("Reading validation dataset........")
x_valid =  pd.read_csv("D:/iNeuron Internship/code/x_valid.csv")
y_valid =  pd.read_csv("D:/iNeuron Internship/code/y_valid.csv")
logging.info("Read the validation data successfully.")


# creating pipelines
pipe4 = Pipeline([("minmax_scalar", MinMaxScaler()), ("XGboost", XGBClassifier())])

pipe6 = Pipeline([("minmax_scalar", MinMaxScaler()), ("random_forest", RandomForestClassifier())])
logging.info("Created pipelines for our models.")

features = ["FTI", "TSH", "TT4", "T4U"] # Permutation method of feature selection was used.
logging.info("Selected top 4 important features.")

# Using randomized search cv to get the best parameter values
logging.info("Hyperparameter tuning intiated in Random Forest.......")
def hyparameter_tuning_rf(model, x, y, final_features):
    params = { 
        'random_forest__max_depth': [15, 25, 30, 35, 45, 50],
        'random_forest__n_estimators': [50, 70, 100, 200, 300, 400]
             }
    tuned_model = RandomizedSearchCV(model, param_distributions=params, n_iter=3, cv=3)
    tuned_model.fit(x[final_features], y)
    logging.info(tuned_model.best_params_)
    return tuned_model

logging.info("Random forest training started......")
model_rf = hyparameter_tuning_rf(pipe6, x_train, y_train, features)
features = ["FTI", "TSH", "TT4", "T4U"] # Permutation method of feature selection was used.
logging.info("Random forest trained.")

# Using randomized search cv to get the best parameter values
logging.info("Hyperparameter tuning intiated in XGBoost.......")
def hyparameter_tuning_xgb(model, x, y, final_features):
    params = { 
       'XGboost__max_depth': [3,4,5,7,10,15,],
       'XGboost__learning_rate': [0.001, 0.0003, 0.005],
       'XGboost__n_estimators': [1000, 1500, 8000, 10000],
       'XGboost__colsample_bytree': [0.3, 0.5, 0.7, 0.9]
             }
    tuned_model = RandomizedSearchCV(model, param_distributions=params, n_iter=3, cv=3)
    tuned_model.fit(x[final_features], y)
    logging.info(tuned_model.best_params_)
    return tuned_model

logging.info("XGBoost training started.")
model_xgb = hyparameter_tuning_xgb(pipe4, x_train, y_train, features)
logging.info("XGBoost trained.")

# Finally saving our model as a pickel file. (For deployment)
pickle.dump(model_rf, open('Random_forest_model.pkl','wb'))
logging.info("Successfully saved Random forest as pickle file.")
pickle.dump(model_rf, open('XGBoost_model.pkl','wb'))
logging.info("Successfully saved XGBoost as pickle file.")
logging.info("Sucessfully executed!")
