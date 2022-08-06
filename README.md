# Two macine learning modules

## First module - trips

### Description

Model suggest the city which should be choosen 
for vaction by the next provided parameters:
 - salary
 - age
 - number of family members
 - city of living
 - preferred type of activity
 - preferred type of transport

## Second module - currency

### Description

The model predicts the dollar exchange rate based on the Central Bank database.
Various algorithms for training models were used.
Their results are displayed on the graph for convenient
comparison with real data and determining the quality of the models.
The mean absolute error for each model is also displayed.
For one of the models, the optimal parameters were selected
by using GridSearch CV (Cross-Validation).

## Run

Both modules are run using these steps:
1. Dependency installation `python -m pip install -r requirements.txt`
2. Run `python trips_model.py`/`python currency.py`
 
## Tools
For this project next tools were used:
 - library [pandas](https://pandas.pydata.org/)
 - library [sklearn](https://scikit-learn.org/stable/)
 - library [matplotlib](https://matplotlib.org/3.3.2/index.html)
 - database [Central Bank](http://www.cbr.ru/currency_base/dynamics/)
 
 
 