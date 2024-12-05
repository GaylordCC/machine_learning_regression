from fastapi import HTTPException
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import os

class ClassificationAlgorithmService:
    def __init__(self) -> None:
        pass

    def handle_classification_image(
        self
    ):
        mnist = fetch_openml('mnist_784', version=1)
        print(mnist.keys())

        # Get mnist data and target
        X, Y = mnist['data'], mnist['target']
        print(X.shape)
        print(Y.shape)
        # Convert X data into numpy array
        digit = X.to_numpy()[0]
        digit_image = digit.reshape(28,28)

        # Generating a plot and storage it
        results_graphics_path = 'results_graphics'
        filenames = []
        plt.imshow(digit_image, cmap='binary')
        file_path = os.path.join(results_graphics_path, f"plot_classification.png")
        plt.savefig(file_path)
        plt.close()  # Cierra el gráfico para liberar memoria
        filenames.append(file_path)

        # Verifying the value teste above
        Y = Y.astype(np.uint8)
        print(Y[0])

        X_train, X_test, Y_train, Y_test = X[:60000], X[60000:], Y[:60000], Y[60000:]
        Y_train_5 = (Y_train == 5)
        Y_test_5 = (Y_test == 5)

        # SGDC Classifier (Algorithm)
        sgd_classifier = SGDClassifier(random_state=42)
        sgd_classifier.fit(X_train, Y_train_5)

        # Check with sgdc that the number above is 5
        print(sgd_classifier.predict([digit]))

        # Measuring model performance
        print(cross_val_score(sgd_classifier, X_train, Y_train_5, cv=3, scoring='accuracy'))

        # Confusion matrix:
        # [
        # TN"True-Negative": algotritmo dijo que no era 5, y estaba correcta, FP"False-Positive": algorimot dijo que eran 5, y se equivoco,
        # FN"False-Negative": algoritmo dijo que no eran 5, y se equivoco, TP"True-Positive": algoritmo dijo que era 5, y estaba correcto
        #]
        Y_train_predict = cross_val_predict(sgd_classifier, X_train, Y_train_5, cv=3)
        print(confusion_matrix(Y_train_5, Y_train_predict))

        # Accuracy: TP/(TP+FP)
        print(precision_score(Y_train_5, Y_train_predict))
        # Memory, recall: TP/(TP+FN)
        print(recall_score(Y_train_5, Y_train_predict))

        # F1 score evaluation 
        print(f1_score(Y_train_5, Y_train_predict))
        return "successfully image classification"
    
    def handle_logistic_classification(
        self
    ):
        # data = pd.read_csv('/home/gaylord/machine_learning_v01/machine_learning/sample_data/Social_Network_Ads.csv') # gecc laptop
        data = pd.read_csv('/mnt/c/Users/Gaylord Carrillo/Documents/develop/machine_learning_regression/machine_learning/sample_data/Social_Network_Ads.csv') # gecc desktop
        
        # Get data information
        print(data.info())
        # Get data description
        print(data.describe())
        # Retrieved column names
        print(data.columns)

        # Extract de column data
        X = data.iloc[:, [2,3]]
        Y = data.iloc[:, -1].values
        print({"Y":Y, "X":X})

        # Handle the categorical variable, and convert this one to numeric column
        gender = data[['Gender']]
        print({"Gender":gender})
        cat_encoder = OneHotEncoder()
        data_cat_1hot = cat_encoder.fit_transform(gender)
        # Check how was the categories
        print(cat_encoder.categories_)
        # Check the conmversion process
        print(data_cat_1hot.toarray()[:3])

        # 
        encoded_df = pd.DataFrame(data_cat_1hot.toarray(), columns = cat_encoder.get_feature_names_out())
        print(encoded_df)
        print('*'*50)

        # Join variables
        data1 = pd.concat([X, encoded_df], axis=1)
        print(data1)
        # Check the quantity of data
        data1.shape
        # Divide the data for training and testing
        X_train, X_test, Y_train, Y_test = train_test_split(data1, Y, test_size=0.2, random_state=0)
        print(X_train.shape)
        print(X_test.shape)

        # Scalate the data, to reduce the big difference between them
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.fit_transform(X_test)
        print(X_train)
        print(X_test)

        # New training of the model
        log_reg = LogisticRegression(random_state=0)
        log_reg.fit(X_train, Y_train)

        # Make some data prediction
        y_pred = log_reg.predict(X_test)
        print('Reales:', Y_test[:15], 'Prediction:', y_pred[:15])

        # Generatin the confusion matrix
        # [
        # TN"True-Negative": algotritmo dijo que no era #, y estaba correcta, FP"False-Positive": algorimot dijo que eran #, y se equivoco,
        # FN"False-Negative": algoritmo dijo que no eran #, y se equivoco, TP"True-Positive": algoritmo dijo que era #, y estaba correcto
        #]
        print(confusion_matrix(Y_test, y_pred))

        # Evaluatind the model performance
        # Accuracy: TP/(TP+FP)
        print('Precision:', precision_score(Y_test, y_pred))
        # Memory, recall: TP/(TP+FN)
        print('Memory:', recall_score(Y_test, y_pred))
        # F1 score evaluation: evaluate accurancy + memory
        print('F1_score:', f1_score(Y_test, y_pred))


        return "successfully logistic classification"
    

    def handle_knn_classification(
        self
    ):
        
        return "successfully knn classification"