# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler 
from sklearn.model_selection import RandomizedSearchCV
import pickle
import logging

logging.basicConfig(filename="Thyroid_api.log", level=logging.INFO)

df =  pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/thyroid0387.data")

df["outcome"] = df["-[840801013]"].str[0]
df.drop(columns="-[840801013]", inplace=True)

# Replacing all possible disease outcomes into one category - "yes".
list = ['S', 'F', 'A', 'R', 'I', 'M', 'N', 'G', 'K', 'L', 'Q', 'J',
       'C', 'O', 'H', 'D', 'P', 'B', 'E']
df['outcome'].replace(to_replace=list, value="yes", inplace=True)

# Replacing the binary outputs into integer values 0 and 1 for simplicity.
df.outcome.replace({"-":0, "yes":1}, inplace=True)

df.rename(columns = {"29":"age", "F":"sex", "f":"thyroxine", "f.1":"query_thyroxine", "f.2":"medication","f.3":"sick", 
                        "f.4":"pregnant", "f.5":"surgery", "f.6":"I131_treatment", "t":"query_hypothyroid", 
                        "f.7":"query_hyperthyroid", "f.8":"lithium", "f.9":"goitre", "f.10":"tumor", "f.11":"hypopituitary", 
                        "f.12":"psych", "t.1":"TSH_measured","0.3":"TSH", "f.13":"T3_measured", "?":"T3", 
                        "f.14":"TT4_measured", "?.1":"TT4", "f.15":"T4U_measured", "?.2":"T4U", "f.16":"FTI_measured", 
                        "?.3":"FTI", "f.17":"TBG_measured", "?.4":"TBG", "other":"referral_source"}, inplace=True)

# We saw that the null values were imputed by a "?". To keep things simple I re-converted the "?" into numpy null value.
df.replace({"?":np.nan}, inplace=True)

# The columns "TBG" and "T3" has a lot of null values. Imputing these null values might reduce variance and might tease 
# the model into giving more importance to a particular case.
df.drop(columns=["TBG", "T3"], inplace=True)

# The column "sex" has a few null values and I thought imputing them with "unknown" makes more sense 
# than male "M" or "female" "F".
df.sex.fillna("unknown", inplace=True)

# Coverting the datatype of continous features into numeric type.
df.TSH = pd.to_numeric(df.TSH)
df.TT4 = pd.to_numeric(df.TT4)
df.T4U = pd.to_numeric(df.T4U)
df.FTI = pd.to_numeric(df.FTI)

index_age = df[df["age"]>100].index
df.drop(index_age, inplace=True)

index_tsh = df[df["TSH"]>15].index
df.drop(index_tsh, inplace=True)

# Encoding the categorical features. 
df_dummy = pd.get_dummies(df)

# Imputing null values using KNNImputer.
def Imputation(df):
    imputer = KNNImputer(n_neighbors=3)
    df_1 = imputer.fit_transform(df)
    df_2 = pd.DataFrame(df_1, columns=df.columns)
    return df_2
    

df_final = Imputation(df_dummy[:7000])

# Splitting the data into train, test and validation to prevent data leakage.
validation_data = df_dummy[7000:]
x_train, x_test, y_train, y_test = train_test_split(df_final.drop(columns="outcome"), df_final["outcome"], test_size=0.2)

valid_final = Imputation(validation_data)

#
def balance_data(x,y):    
    ros = RandomOverSampler(random_state=42)
    x_sample, y_sample = ros.fit_resample(x, y)
    return x_sample, y_sample

x_train, y_train = balance_data(x_train, y_train)
x_test, y_test = balance_data(x_test, y_test)
x_valid, y_valid = balance_data(valid_final.drop(columns="outcome"), valid_final["outcome"])


pipe4 = Pipeline([("minmax_scalar", MinMaxScaler()), ("XGboost", XGBClassifier())])

pipe6 = Pipeline([("minmax_scalar", MinMaxScaler()), ("random_forest", RandomForestClassifier())])

# Using randomized search cv to get the best parameter values
def hyparameter_tuning_rf(model, x, y, final_features):
    params = { 
        'random_forest__max_depth': [15, 25, 30, 35, 45, 50],
        'random_forest__n_estimators': [50, 70, 100, 200, 300, 400]
             }
    tuned_model = RandomizedSearchCV(model, param_distributions=params, n_iter=3, cv=3)
    tuned_model.fit(x[final_features], y)
    logging.info(tuned_model.best_params_)
    return tuned_model

features = ["FTI", "TSH", "TT4", "T4U"] # Permutation method of feature selection was used.

model_rf = hyparameter_tuning_rf(pipe6, x_train, y_train, features)

# Using randomized search cv to get the best parameter values
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

model_xgb = hyparameter_tuning_xgb(pipe4, x_train, y_train, features)

# Finally saving our model as a pickel file. (For deployment)
pickle.dump(model_rf, open('Random_forest_model.pkl','wb'))
pickle.dump(model_xgb, open('XGBoost_model.pkl','wb'))
logging.info("Sucessfully executed!")
