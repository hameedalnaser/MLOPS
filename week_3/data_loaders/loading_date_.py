if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from pathlib import Path
import pandas as pd

def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df

@data_loader
def load_data(*args, **kwargs):
    """
    Loads NYC Yellow Taxi data for March 2023.
    """
    data = read_dataframe(2023, 3)
    print(data.shape)
    return data

@test
def test_output(output, *args) -> None:
    """
    Basic test to ensure data was loaded.
    """
    assert output is not None, 'The output is undefined'
    assert not output.empty, 'The loaded dataframe is empty'
