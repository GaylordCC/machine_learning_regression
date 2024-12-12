from fastapi import HTTPException
from ..schemas import RegressionSchema
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, OrdinalEncoder, OneHotEncoder

from sklearn.svm import SVR
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

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
            plt.figure()  # create a new figure
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
        
    def tree_regression(
        self
    ):
        try:
            # Get the data in the folder path
            data = pd.read_csv('/home/gaylord/machine_learning_v01/machine_learning/sample_data/housing.csv') # gecc laptop
            # data = pd.read_csv('/mnt/c/Users/Gaylord Carrillo/Documents/develop/machine_learning_regression/machine_learning/sample_data/housing.csv') # gecc desktop 
            
            # Filter the data to be used in the corralation analysis
            data_for_corr = data.drop(['ocean_proximity'], axis=1)
            # corr_matrix = data_for_corr.corr()
            # corr_matrix['median_house_value'].sort_values(ascending=False)
            # print(corr_matrix)
            # print('#'*50)
            
            # Get the data information
            data.info()
            
            # Attribute combinations
            data_for_corr['rooms_per_household'] = data_for_corr['total_rooms'] / data_for_corr['households']
            data_for_corr['bedrooms_per_room'] = data_for_corr['total_bedrooms'] / data_for_corr['households']
            data_for_corr['population_per_household'] = data_for_corr['population'] / data_for_corr['households']
            corr_matrix = data_for_corr.corr()
            corr_matrix['median_house_value'].sort_values(ascending=False)
            print(corr_matrix)
            print('#'*50)

            # Clean data and handle categorical attributes
            # Fill lack data
            data_for_corr['total_bedrooms'].fillna(data_for_corr['total_bedrooms'].median(), inplace=True)
            data_for_corr.info()
            print(data_for_corr)
            print('X'*50)

            # Manipulation of category data
            data_ocean = data[['ocean_proximity']]
            Ordinal_encoder = OrdinalEncoder()
            data_ocean_encoder = Ordinal_encoder.fit_transform(data_ocean)
            np.random.choice(data_ocean_encoder.ravel(), size=10)
            # Manipulation of category data
            cat_ecoder = OneHotEncoder()
            data_cat_1hot = cat_ecoder.fit_transform(data_ocean)
            print(data_cat_1hot.toarray())
            print('Y'*50)
            print(cat_ecoder.categories_)
            # DataFrame with the categorical variables
            encoded_df = pd.DataFrame(data_cat_1hot.toarray(), columns= cat_ecoder.get_feature_names_out())
            print(encoded_df.head())

            # Creating machine learning (ML) algorith
            # y = data_for_corr['median_house_value'].values.reshape(-1,1)

            # X = data_for_corr[[
            #     'median_income',
            #     'rooms_per_household',
            #     'total_rooms',
            #     'housing_median_age',
            #     'households'
            # ]]

            # # Concatenate categorical variables
            # data1 = pd.concat([X, encoded_df], axis=1)
            # data1.columns
            # X = data1.values
            # X[:10]
            
            # # Multi Linear Regression
            # X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
            # lin_reg = LinearRegression()
            # # Training
            # lin_reg.fit(X_train, Y_train)
            # # Prediction
            # y_pred = lin_reg.predict(X_test)
            # # Determine r2
            # r2 = r2_score(Y_test, y_pred)

            # Scalate the independent variable
            # sc_X = StandardScaler()
            # X = sc_X.fit_transform(X)

            # Multi Linear Regression
            # X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
            # lin_reg = LinearRegression()
            # # Training
            # lin_reg.fit(X_train, Y_train)
            # # Prediction
            # y_pred = lin_reg.predict(X_test)
            # # Determine r2
            # r2 = r2_score(Y_test, y_pred)


            #
            columnas = ['median_income','rooms_per_household','total_rooms','housing_median_age','households','latitude','longitude']
            col_modelo = []
            y = data_for_corr['median_house_value'].values.reshape(-1,1)
            
            for col in columnas:
                col_modelo.append(col)
                data1 = data_for_corr[col_modelo]
                data1 = pd.concat([data1, encoded_df], axis=1)
                X = data1.values
                X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
                lin_reg = LinearRegression()
                # Training
                lin_reg.fit(X_train, Y_train)
                # Prediction
                y_pred = lin_reg.predict(X_test)
                # Determine r2
                r2 = r2_score(Y_test, y_pred)
                print('Columnas:', col_modelo, 'Calificacion', r2)


            # Create the histograms
            histograms = data.hist(bins=50, figsize=(20,15))
            print(data)

            # Path to save the graphical results
            results_graphics_path = 'results_graphics'

            # Generate scatter plot
            scatter_plot_path = os.path.join(results_graphics_path, "scatter_plot.png")
            data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=data['population']/100, label='population', figsize=(15,7), c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
            plt.legend()
            plt.savefig(scatter_plot_path)  # Save the scatter plot
            plt.close()  # Close the scatter plot to free up memory

            # Generate correlation plot by heatmap
            correlation_plot_path = os.path.join(results_graphics_path, "correlation_plot.png")
            plt.figure(figsize=(20,10))
            sns.heatmap(data_for_corr.corr(), annot=True)
            plt.savefig(correlation_plot_path)  # Save the heatmap plot
            plt.close()  # Close the scatter plot to free up memory


            # Generate histograms plot
            file_path = os.path.join(results_graphics_path, "histograms.png")
            plt.savefig(file_path)  # Save the entire figure generated by data.hist()
            plt.close('all')  # Close all figures to free up memory
            return "Histograms generated successfully"
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f" tree regression: {str(e)}"
            )
        
    def random_tree_regression(
        self      
    ):
        try:
            # Get the data in the folder path
            data = pd.read_csv('/home/gaylord/machine_learning_v01/machine_learning/sample_data/housing.csv') # gecc laptop
            # data = pd.read_csv('/mnt/c/Users/Gaylord Carrillo/Documents/develop/machine_learning_regression/machine_learning/sample_data/housing.csv') # gecc desktop 
            # Filter the data to be used in the corralation analysis
            data_for_corr = data.drop(['ocean_proximity'], axis=1)
            
            # Attribute combinations
            data_for_corr['rooms_per_household'] = data_for_corr['total_rooms'] / data_for_corr['households']
            data_for_corr['bedrooms_per_room'] = data_for_corr['total_bedrooms'] / data_for_corr['households']
            data_for_corr['population_per_household'] = data_for_corr['population'] / data_for_corr['households']
            corr_matrix = data_for_corr.corr()
            corr_matrix['median_house_value'].sort_values(ascending=False)
            print(corr_matrix)
            print('#'*50)

            # Clean data and handle categorical attributes
            # Fill lack data
            data_for_corr['total_bedrooms'].fillna(data_for_corr['total_bedrooms'].median(), inplace=True)
            data_for_corr.info()
            print(data_for_corr)
            print('X'*50)
            # Manipulation of category data
            data_ocean = data[['ocean_proximity']]
            Ordinal_encoder = OrdinalEncoder()
            data_ocean_encoder = Ordinal_encoder.fit_transform(data_ocean)
            np.random.choice(data_ocean_encoder.ravel(), size=10)
            # Manipulation of category data
            cat_ecoder = OneHotEncoder()
            data_cat_1hot = cat_ecoder.fit_transform(data_ocean)
            print(data_cat_1hot.toarray())
            print('Y'*50)
            print(cat_ecoder.categories_)
            # DataFrame with the categorical variables
            encoded_df = pd.DataFrame(data_cat_1hot.toarray(), columns= cat_ecoder.get_feature_names_out())
            #
            columnas = ['median_income','rooms_per_household','total_rooms','housing_median_age','households','latitude','longitude']
            col_modelo = []
            y = data_for_corr['median_house_value'].values.reshape(-1,1)
            
            for col in columnas:
                col_modelo.append(col)
                data1 = data_for_corr[col_modelo]
                data1 = pd.concat([data1, encoded_df], axis=1)
                X = data1.values
                X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
                tree_reg = DecisionTreeRegressor()
                # Training
                tree_reg.fit(X_train, Y_train)
                # Prediction
                y_pred = tree_reg.predict(X_test)
                # Determine r2
                r2 = r2_score(Y_test, y_pred)
                print('Columnas:', col_modelo, 'Calificacion', r2)
            
            return "test random tree regression"
        except Exception as e:
            raise HTTPException (
                status_code=422,
                detail=f"Random tree regression: {(e)}"
            )
        
    def random_forest_regression(
            self
    ):
        try:
            # Get the data in the folder path
            data = pd.read_csv('/home/gaylord/machine_learning_v01/machine_learning/sample_data/housing.csv') # gecc laptop
            # data = pd.read_csv('/mnt/c/Users/Gaylord Carrillo/Documents/develop/machine_learning_regression/machine_learning/sample_data/housing.csv') # gecc desktop 
            # Filter the data to be used in the corralation analysis
            data_for_corr = data.drop(['ocean_proximity'], axis=1)
            
            # Attribute combinations
            data_for_corr['rooms_per_household'] = data_for_corr['total_rooms'] / data_for_corr['households']
            data_for_corr['bedrooms_per_room'] = data_for_corr['total_bedrooms'] / data_for_corr['households']
            data_for_corr['population_per_household'] = data_for_corr['population'] / data_for_corr['households']
            corr_matrix = data_for_corr.corr()
            corr_matrix['median_house_value'].sort_values(ascending=False)
            print(corr_matrix)
            print('#'*50)

            # Clean data and handle categorical attributes
            # Fill lack data
            data_for_corr['total_bedrooms'].fillna(data_for_corr['total_bedrooms'].median(), inplace=True)
            data_for_corr.info()
            print(data_for_corr)
            print('X'*50)
            # Manipulation of category data
            data_ocean = data[['ocean_proximity']]
            Ordinal_encoder = OrdinalEncoder()
            data_ocean_encoder = Ordinal_encoder.fit_transform(data_ocean)
            np.random.choice(data_ocean_encoder.ravel(), size=10)
            # Manipulation of category data
            cat_ecoder = OneHotEncoder()
            data_cat_1hot = cat_ecoder.fit_transform(data_ocean)
            print(data_cat_1hot.toarray())
            print('Y'*50)
            print(cat_ecoder.categories_)
            # DataFrame with the categorical variables
            encoded_df = pd.DataFrame(data_cat_1hot.toarray(), columns= cat_ecoder.get_feature_names_out())
            #
            columnas = ['median_income','rooms_per_household','total_rooms','housing_median_age','households','latitude','longitude']
            col_modelo = []
            y = data_for_corr['median_house_value'].values
            
            for col in columnas:
                col_modelo.append(col)
                data1 = data_for_corr[col_modelo]
                data1 = pd.concat([data1, encoded_df], axis=1)
                X = data1.values
                X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
                forest_reg = RandomForestRegressor()
                # Training
                forest_reg.fit(X_train, Y_train)
                # Prediction
                y_pred = forest_reg.predict(X_test)
                # Determine r2
                r2 = r2_score(Y_test, y_pred)
                print('Columnas:', col_modelo, 'Calificacion', r2)
            return "Test random forest regression"
        except Exception as e:
            raise HTTPException (
                status_code=422,
                detail=f"Random forest regression: {(e)}"
            )