import pandas
import matplotlib.pyplot as plt
from numpy import linspace
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

usd_rate = pandas.read_excel('currency_table.xlsx')
curs = usd_rate.curs

# X - Входные данные, на основе чего делаем прогноз
# y - Выходные данные, то что прогнозируем

FUTURE_DAYS = 7
PAST_DAYS = 14
start = PAST_DAYS
end = len(curs) - FUTURE_DAYS
total = end - start

past_X = []
future_y = []

for i in range(start, end):
    X = curs[i - PAST_DAYS:i]
    past_X.append(list(X))
    y = curs[i:i + FUTURE_DAYS]
    future_y.append(list(y))

past_columns = [f'past_{i}' for i in range(PAST_DAYS)]
future_columns = [f'future_{i}' for i in range(FUTURE_DAYS)]

df_X = pandas.DataFrame(data=past_X, columns=past_columns)
df_y = pandas.DataFrame(data=future_y, columns=future_columns)

# обучающая выборка (тренировочная)
X_train = df_X[:-10]
y_train = df_y[:-10]

# тестовая выборка (проверочная)
X_test = df_X[-10:]
y_test = df_y[-10:]

# Регрессия - прогнозирование конкретного значения величины (не дискретной)
# Классификация - предсказание принадлежности к ограниченному количеству классов (дискретное значение)

# Обучаем модель и делаем предсказание
forest = RandomForestRegressor()
forest.fit(X_train, y_train)
prediction = forest.predict(X_test)

plt.plot(y_test.iloc[0], label='Real data')
plt.plot(prediction[0], label='Prediction')

mae = mean_absolute_error(y_test.iloc[0], prediction[0])
print(f'mean_absolute_error = {mae}')

# Вариант с другим алгоритмом (KNeighborsRegressor)
knn = KNeighborsRegressor()
knn.fit(X_train, y_train)
knn_prediction = knn.predict(X_test)
plt.plot(knn_prediction[0], label='KNN Prediction')
knn_mae = mean_absolute_error(y_test.iloc[0], knn_prediction[0])
print(f'KNN mean_absolute_error = {knn_mae}')

# Вариант с другим алгоритмом (MLPRegressor)
mlp = MLPRegressor()
mlp.fit(X_train, y_train)
mlp_prediction = mlp.predict(X_test)
plt.plot(mlp_prediction[0], label='MLP Prediction')
mlp_mae = mean_absolute_error(y_test.iloc[0], mlp_prediction[0])
print(f'MLP mean_absolute_error = {mlp_mae}')


# GridSearch CV (Cross-Validation)
model = MLPRegressor(random_state=42)
param_grid = {
    'max_iter': [500, 1000, 2000],
    'learning_rate_init': linspace(0.001, 0.01, 5),
    'activation': ['logistic', 'relu']
}
GS = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', cv=3)
GS.fit(X_train, y_train)
best_params = GS.best_params_
best_model = GS.best_estimator_
print(f'best_params = {best_params}, best_model = {best_model}')

best_mlp = best_model
best_mlp.fit(X_train, y_train)
best_mlp_prediction = best_mlp.predict(X_test)
plt.plot(best_mlp_prediction[0], label='Best MLP Prediction')
best_mlp_mae = mean_absolute_error(y_test.iloc[0], best_mlp_prediction[0])
print(f'Best MLP mean_absolute_error = {best_mlp_mae}')

# Вывод графика
plt.legend()
plt.show()





