import pandas as pd
import numpy as np
import sklearn
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder, OrdinalEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import median_absolute_error, mean_absolute_error, mean_absolute_percentage_error, make_scorer

print('import fatti')
df = pd.read_csv('Smart_Farming_Crop_Yield_2024.csv')

print('dataset importato')

df['crop_disease_status'] = df['crop_disease_status'].fillna('None')
df['irrigation_type'] = df['irrigation_type'].fillna('None')


def ciclic_encoding(df, col, max_val):

  #result[date_col] = pd.to_datetime(result[date_col])


  df[col] = pd.to_datetime(df[col])
  month = df[col].dt.month


  df[col + '_sin'] = np.sin(2 * np.pi * month/max_val)
  df[col + '_cos'] = np.cos(2 * np.pi * month/max_val)


  return df

df = ciclic_encoding(df, 'harvest_date', 12)
df = ciclic_encoding(df, 'sowing_date', 12)

df.drop(['farm_id', 'sensor_id', 'timestamp', 'latitude', 'longitude', 'sowing_date', 'harvest_date'], axis=1, inplace=True)

print('dataset preprocessato')

x = df.drop(columns=['yield_kg_per_hectare'])
y = df['yield_kg_per_hectare']

# per ricevere in input un dataframe e dare in output un dataframe
sklearn.set_config(transform_output='pandas')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print('dataset splittato')

categories_list = [
    ['Central USA', 'East Africa', 'North India', 'South USA', 'South India'],  # region
    ['Maize', 'Soybean', 'Cotton', 'Wheat', 'Rice'],                           # crop_type
    ['None', 'Sprinkler', 'Manual', 'Drip'],                                  # irrigation_type
    ['Inorganic', 'Mixed', 'Organic'],                                          # fertilizer_type
    ['Severe', 'None', 'Mild', 'Moderate']                                    # crop_disease_status
]

"""# GRID SEARCH"""

encoder = ColumnTransformer(
    [
        ('encoder', 'passthrough', ['region', 'crop_type', 'irrigation_type', 'fertilizer_type', 'crop_disease_status'])
    ],
    remainder='passthrough',
    verbose_feature_names_out=False,
    force_int_remainder_cols=False
)

print('encoder creato')

pipe = Pipeline([
    ('encoder', encoder),
#    ('standardization', StandardScaler()),
    ('regressor', RandomForestRegressor())
])

print('pipe creata')

pipe.get_params()

"""
Parametri Grid"""

params = [
    {
        'encoder__encoder': [OneHotEncoder(categories = categories_list, sparse_output = False, drop = "first")],
        #'encoder__encoder': [OneHotEncoder(sparse_output=False, drop='first')],
        'regressor__n_estimators' : [100, 150, 200],
        'regressor__criterion' : ['squared_error', 'absolute_error'],
        'regressor__max_depth' : [3, 4, 5, 6, 7, 9]
    },
    {
        'encoder__encoder': [OrdinalEncoder()],
        'regressor__n_estimators' : [100, 150, 200],
        'regressor__criterion' : ['squared_error', 'absolute_error'],
        'regressor__max_depth' : [3, 4, 5, 6, 7, 9,]
    },
]

print('parametri impostati')

grid_search = GridSearchCV(
    estimator = pipe,
    param_grid= params,
    scoring= make_scorer(mean_absolute_error, greater_is_better = False),
    #n_jobs = -1,
    cv = KFold(n_splits = 5, shuffle = True, random_state = 42),
    refit = True, # dopo aver controllato tutti gli allenamenti, seleziona il migliore in base alla metrica scelta
    verbose = 4
)

print('grid search creata')

grid_search.fit(x_train, y_train)

print('grid search fittata')

"""Risultati con encoder passthrough"""

print(grid_search.best_params_)
#{'encoder__encoder': OneHotEncoder(drop='first', sparse_output=False),
# 'regressor__criterion': 'squared_error',
# 'regressor__max_depth': 9,
# 'regressor__n_estimators': 100}

print(grid_search.best_score_)
# np.float64(-1009.2749072662275)

best_grid = grid_search.best_estimator_

y_test_pred = best_grid.predict(x_test)

print('il mae è: '+str(mean_absolute_error(y_test, y_test_pred)))
# 1039.7284892361877

print('il mape è: '+str(mean_absolute_percentage_error(y_test, y_test_pred)))

# 0.2960973580960324


joblib.dump(best_grid, 'best_grid.joblib')
#best_grid = joblib.load('best_grid.joblib')

print('modello salvato')
print('fine del test')