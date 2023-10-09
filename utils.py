import pandas as pd

def load_data_from_csv(filepath):
    """ Load data from a CSV file into a DataFrame. """
    return pd.read_csv(filepath)

def save_data_to_csv(data, filepath):
    """ Save a DataFrame to a CSV file. """
    data.to_csv(filepath)

def round_dataframe(data, decimals=2):
    """ Round all numerical values in a DataFrame. """
    return data.round(decimals)

def resample_to_business_days(data):
    """ Resample a DataFrame indexed by date to business days. """
    resampled_data = data.resample('B').ffill().bfill()
    return resampled_data[resampled_data.index.isin(data.index) | 
                          (resampled_data.index >= data.index.min()) & 
                          (resampled_data.index <= data.index.max())]


def calculate_mae(true_values, predicted_values):
    """ Calculate the Mean Absolute Error (MAE). """
    return (true_values - predicted_values).abs().mean()
