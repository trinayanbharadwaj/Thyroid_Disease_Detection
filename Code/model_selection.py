# Importing the necessary libraries.
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score
from tensorflow import keras
from tensorflow.keras import layers
import logging

logging.basicConfig(filename="Model_selection.log", level=logging.INFO, format='%(asctime)s %(message)s',
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

logging.info("Creating the pipelines.....")
# Creating pipelies.
pipe1 = Pipeline([("minmax_scalar", MinMaxScaler()), ("logistic_regression", LogisticRegression())])

pipe2 = Pipeline([("minmax_scalar", MinMaxScaler()), ("KNN", KNeighborsClassifier())])

pipe3 = Pipeline([("minmax_scalar", MinMaxScaler()), ("svm", SVC())])

pipe4 = Pipeline([("minmax_scalar", MinMaxScaler()), ("XGboost", XGBClassifier())])

pipe5 = Pipeline([("minmax_scalar", MinMaxScaler()), ("decision_tree", DecisionTreeClassifier())])

pipe6 = Pipeline([("minmax_scalar", MinMaxScaler()), ("random_forest", RandomForestClassifier())])
logging.info("Pipelines created.")

logging.info("Builiding an ANN model....")
def build_ann():
    
    model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[54]),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')])
    
    model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["binary_accuracy"])
    
    return model

ann_model = build_ann()
logging.info("ANN built and compiled.")

logging.info("Fitting the pipelines and ANN")
# Fitting the pipelines
pipelines = [pipe1, pipe2, pipe3, pipe4, pipe5, pipe6]

for pipe in pipelines:
    pipe.fit(x_train, y_train)
logging.info("All the pipelines fitted.")    
    
callback = keras.callbacks.EarlyStopping(monitor = "val_binary_accuracy", patience=3, restore_best_weights=True)

history = ann_model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=32,
    epochs=100, 
    callbacks = [callback])
logging.info("ANN trained and validated")

logging.info("Predicting on validation data")
# Predicting
pred1 = pipe1.predict(x_valid)
pred2 = pipe2.predict(x_valid)
pred3 = pipe3.predict(x_valid)
pred4 = pipe4.predict(x_valid)
pred5 = pipe5.predict(x_valid)
pred6 = pipe6.predict(x_valid)
logging.info("Prediction done.")

logging.info("Displaying the performance metrics of all the model.......")
# Comparing the result of each pipeline and selecting the best pipeline. 
logging.info("Accuracy of Logistic_Regression {}" .format(round(accuracy_score(y_valid, pred1)*100, 2)))
logging.info("Recall of Logistic_Regression {}" .format(round(recall_score(y_valid, pred1),2)))
logging.info("===================================================================")
logging.info("Accuracy of KNN {}" .format(round(accuracy_score(y_valid, pred2)*100, 2)))
logging.info("Recall of KNN {}" .format(round(recall_score(y_valid, pred2),2)))
logging.info("===================================================================")
logging.info("Accuracy of SVC {}" .format(round(accuracy_score(y_valid, pred3)*100,2)))
logging.info("Recall of SVC {}" .format(round(recall_score(y_valid, pred3),2)))
logging.info("===================================================================")
logging.info("Accuracy of xgboost {}" .format(round(accuracy_score(y_valid, pred4)*100,2)))
logging.info("Recall of xgboost {}" .format(round(recall_score(y_valid, pred4),2)))
logging.info("===================================================================")
logging.info("Accuracy of decision_tree {}" .format(round(accuracy_score(y_valid, pred5)*100,2)))
logging.info("Recall of decision_tree {}" .format(round(recall_score(y_valid, pred5),2)))
logging.info("===================================================================")
logging.info("Accuracy of Random_forest {}" .format(round(accuracy_score(y_valid, pred6)*100,2)))
logging.info("Recall of Random_forest {}" .format(round(recall_score(y_valid, pred6),2)))
logging.info("Select the best performing model and Model Selection phase is completed.")
