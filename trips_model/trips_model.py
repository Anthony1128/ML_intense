import pandas
from sklearn.ensemble import RandomForestClassifier

# Prepare dataset
trips = pandas.read_excel('trips_data.xlsx', index_col=0)
df = pandas.get_dummies(trips, columns=['city', 'vacation_preference', 'transport_preference'])

# Base value for prediction
X = df.drop('target', axis=1)

# Value that is going to be predicted
y = df['target']

# Create and teach model
model = RandomForestClassifier()
model.fit(X, y)

# Chech on test example
data_example = {i: [0] for i in X.columns}
# print(data_example)
example = {'salary': [120000],
           'age': [35],
           'family_members': [1],
           'city_Екатеринбург': [0],
           'city_Киев': [0],
           'city_Краснодар': [0],
           'city_Минск': [0],
           'city_Москва': [0],
           'city_Новосибирск': [0],
           'city_Омск': [0],
           'city_Петербург': [0],
           'city_Томск': [0],
           'city_Хабаровск': [1],
           'city_Ярославль': [0],
           'vacation_preference_Архитектура': [1],
           'vacation_preference_Ночные клубы': [0],
           'vacation_preference_Пляжный отдых': [0],
           'vacation_preference_Шоппинг': [0],
           'transport_preference_Автомобиль': [0],
           'transport_preference_Космический корабль': [0],
           'transport_preference_Морской транспорт': [0],
           'transport_preference_Поезд': [0],
           'transport_preference_Самолет': [1]}
df_example = pandas.DataFrame(example)
print(model.predict(df_example))

