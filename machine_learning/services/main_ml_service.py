from fastapi import HTTPException
from ..schemas import RegressionSchema
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVR
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

            # Split the data sample into trainind and testing data
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
            Y = data['Sales'].values                              # Sales data

            # Split the data sample into trainind and testing data
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
            # Data to be use in the polynomical regression
            pos = [x for x in range(1,11)]
            post = [
                "Pasante de Desarrollo",
                "Desarrollador Junior",
                "Desarrollador Intermedio",
                "Desaarrollador Senior",
                "Lider de Proyecto",
                "Gerente de Proyecto",
                "Arquitecto de Software",
                "Director de Desarrollo",
                "Director de Tecnologia",
                "Director Ejecutivo (CEO)",
            ]
            salary = [1200, 2500, 4000, 4800, 6500, 9000, 12850, 15000, 25000, 50000]
            data = {
                "position": post,
                "years": pos,
                "salary": salary
            }
            data = pd.DataFrame(data)
            data.head()
            print(data)
            # Convert DataFrame to a list of dictionaries
            data_dict = data.to_dict(orient="records")

            X = data.iloc[:, 1].values.reshape(-1,1)
            Y = data.iloc[:, -1].values
            
            # Linear regression test
            regression = LinearRegression()
            regression.fit(X,Y)

            # Polynomical regression test
            poly = PolynomialFeatures(degree=4)
            X_poly = poly.fit_transform(X)
            print(X_poly)
            regression_poly = LinearRegression()
            regression_poly.fit(X_poly, Y)
            # Predict a specific value
            predic_v = poly.fit_transform([[2]])
            print(regression_poly.predict(predic_v))

            # Determine R2
            y_pred = regression_poly.predict(X_poly)
            print(r2_score(Y, y_pred))


            results_graphics_path = 'results_graphics'
            filenames = []
            plt.figure()  # Crear una nueva figura
            plt.scatter(data['years'], data['salary'])
            # plt.plot(X, regression.predict(X), color="black")
            plt.plot(X, regression_poly.predict(X_poly), color="black")
            file_path = os.path.join(results_graphics_path, "polynomicalregression.png")
            plt.savefig(file_path)
            plt.close()
            filenames.append(file_path)
            return data_dict
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"polynomical regression: {str(e)}"
            )
        
    def svr_regression(
            self
        ):
        # Support Vector Regression (SVR)
        # Support Vector Machine
        try:
            # Get the data in the folder path
            # data = pd.read_csv('/home/gaylord/machine_learning_v01/machine_learning/sample_data/Advertising.csv') # gecc laptop 
            data = pd.read_csv('/mnt/c/Users/Gaylord Carrillo/Documents/develop/machine_learning_regression/machine_learning/sample_data/Advertising.csv') # gecc desktop 
            data = data.iloc[:, 1:]
            # TV and Radio data
            X = data.drop(['Newspaper', 'Sales'],axis=1).values
            # Sales data
            Y = data['Sales'].values
            # Split the data sample into trainind and testing data
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            # Intantize
            svr = SVR(kernel='rbf')
            # Perform the training
            svr.fit(X_train, Y_train)
            #Get the predictions
            y_predict = svr.predict(X_test)
            print("Predicciones: {}, Reales: {}".format(y_predict[:4], Y_test[:4]))
            # Evaulate the model by determine r2_score
            print(r2_score(Y_test, y_predict))
            print("*"*10)
            # Convert DataFrame to a list of dictionaries
            data_dict = data.to_dict(orient="records")

            return data_dict
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"support vector regression: {str(e)}"
            )