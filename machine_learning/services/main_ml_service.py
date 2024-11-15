from fastapi import HTTPException
from ..schemas import RegressionSchema
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import os

class MachineLearningService:
    def __init__(self):
        pass

    def handle_user_query(self):
        try:
            # Get the data in the folder path
            # data = pd.read_csv('/home/gaylord/machine_learning_v01/machine_learning/sample_data/Advertising.csv') # gecc laptop 
            data = pd.read_csv('/mnt/c/Users/Gaylord Carrillo/Documents/develop/machine_learning_regression/machine_learning/sample_data/Advertising.csv') # gecc desktop

            # Convert DataFrame to a list of dictionaries
            data_dict = data.to_dict(orient="records")
            # Get data information
            print(data.info())
            # Get data description
            print(data.describe())
            # Retrieved column name
            print(data.columns)

            cols = ['TV', 'Radio', 'Newspaper']
            filenames = []

            results_graphics_path = 'results_graphics'

            for col in cols:
                plt.plot(data[col], data['Sales'], 'ro')
                plt.title(f'Ventas respecto a la publicidad en {col}')
                file_path = os.path.join(results_graphics_path, f"plot_{col}.png")
                plt.savefig(file_path)
                plt.close()  # Cierra el gráfico para liberar memoria
                filenames.append(file_path)


            return data_dict
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"handle_user_query: {str(e)}"
            )
        
    
    def regression_linear_model(
            self,
            request: RegressionSchema
        ):

        try:
            # Get the data in the folder path
            # data = pd.read_csv('/home/gaylord/machine_learning_v01/machine_learning/sample_data/Advertising.csv') # gecc laptop 
            data = pd.read_csv('/mnt/c/Users/Gaylord Carrillo/Documents/develop/machine_learning_regression/machine_learning/sample_data/Advertising.csv') # gecc desktop 
            X = data[request.column_name].values.reshape(-1,1)   # Reshape 'TV', 'Radio', 'Newspaper' column values into a 2D array
            Y = data['Sales'].values

            # Split the data sample into trainind ans testing data
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            print(X_train.shape)
            print(X_test.shape)

            lin_reg = LinearRegression()
            lin_reg.fit(X_train, Y_train)

            y_predict = lin_reg.predict(X_test)

            print("Predicciones: {}, Reales: {}".format(y_predict[:4], Y_test[:4]))

            # Evaluating the regression model
            #RMSE
            rmse = mean_squared_error(Y_test, y_predict, squared=False)
            print('RMSE:', rmse)
            # R2
            r2 = r2_score(Y_test, y_predict)
            print('R2:', r2)

            filenames = []
            results_graphics_path = 'results_graphics'
            plt.plot(X_test, Y_test, 'ro')
            plt.title(f"Regresion lineal para predecir datos de {request.column_name.value}")
            plt.plot(X_test,y_predict)
            file_path = os.path.join(results_graphics_path, f"plotregression.png")
            plt.savefig(file_path)
            plt.close()  # Cierra el gráfico para liberar memoria
            filenames.append(file_path)

            # Convert the NumPy array to a list before returning
            return y_predict.tolist() # Return the prediction as a list
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"regression linear model: {str(e)}"
            )
        
    def regression_multi_linear_model(
            self,
        ):
        try:
            # Get the data in the folder path
            # data = pd.read_csv('/home/gaylord/machine_learning_v01/machine_learning/sample_data/Advertising.csv') # gecc laptop 
            data = pd.read_csv('/mnt/c/Users/Gaylord Carrillo/Documents/develop/machine_learning_regression/machine_learning/sample_data/Advertising.csv') # gecc desktop 
            X = data.drop(['Newspaper', 'Sales'],axis=1).values   # Array of TV and newspaper data
            Y = data['Sales'].values                          # Sales data

            # Split the data sample into trainind ans testing data
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            print(X_train.shape)
            print(X_test.shape)

            lin_reg = LinearRegression()
            lin_reg.fit(X_train, Y_train)

            y_predict = lin_reg.predict(X_test)

            print("Predicciones: {}, Reales: {}".format(y_predict[:4], Y_test[:4]))

            # Evaluating the regression model
            #RMSE
            rmse = mean_squared_error(Y_test, y_predict, squared=False)
            print('RMSE:', rmse)
            # R2
            r2 = r2_score(Y_test, y_predict)
            print('R2:', r2)

            results_graphics_path = 'results_graphics'
            filenames = []
            plt.figure()  # Crear una nueva figura
            sns.regplot(x = Y_test, y = y_predict)
            file_path = os.path.join(results_graphics_path, "plotmultiregression.png")
            plt.savefig(file_path)
            plt.close()
            filenames.append(file_path)
            # Convert the NumPy array to a list before returning
            return y_predict.tolist() # Return the prediction as a list
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"regression multi linear model: {str(e)}"
            )
        
    def polynomical_regression(self):
        try:
            
            return "Test"
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"polynomical regression: {str(e)}"
            )