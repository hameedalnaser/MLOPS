if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

@transformer
def transform(data, *args, **kwargs):
    # Use only the relevant columns
    df = data.copy()
    df = df[['PULocationID', 'DOLocationID', 'duration']]

    # Convert IDs to string (if not already)
    df['PULocationID'] = df['PULocationID'].astype(str)
    df['DOLocationID'] = df['DOLocationID'].astype(str)

    # Prepare training data
    train_dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df['duration'].values

    # Train linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Print model intercept
    print('Intercept:', lr.intercept_)

    # Optional: print RMSE on training data
    y_pred = lr.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    print('RMSE:', rmse)

    # Return model and vectorizer for future use
    return {
        'model': lr,
        'dv': dv
    }

@test
def test_output(output, *args) -> None:
    assert 'model' in output, 'Model is missing'
    assert 'dv' in output, 'DictVectorizer is missing'
