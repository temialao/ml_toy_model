import pandas as pd
from utils import load_data_from_csv, round_dataframe, resample_to_business_days, save_data_to_csv, calculate_mae
import numpy as np

def test_load_data_from_csv():
    # For this, you'd ideally use a small test CSV file.
    df = load_data_from_csv('test_data.csv')
    assert isinstance(df, pd.DataFrame), "Should return a DataFrame"

def test_save_data_to_csv(tmpdir):
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    filepath = tmpdir.join("test_output.csv")
    save_data_to_csv(df, filepath)
    assert filepath.check(), "File should exist"
    # Additional checks can be added to ensure data integrity.

def test_round_dataframe():
    df = pd.DataFrame({
        'A': [1.111, 2.222, 3.333],
        'B': [4.444, 5.555, 6.666]
    })
    rounded_df = round_dataframe(df)
    assert rounded_df.iloc[0, 0] == 1.11, "Value should be rounded to two decimal places"

def test_resample_to_business_days():
    # Create a dummy DataFrame
    data = {
        'value': [1, 2, 3, 4]
    }
    dates = pd.to_datetime(['2023-10-01', '2023-10-02', '2023-10-04', '2023-10-07'])
    df = pd.DataFrame(data, index=dates)

    # Use the function
    resampled_df = resample_to_business_days(df)

    # Check if the data was resampled to include all business days
    expected_dates = pd.date_range('2023-10-01', '2023-10-07', freq='B')
    
    assert len(resampled_df.index) == len(expected_dates)  # Check the lengths first
    assert all(resampled_df.index == expected_dates)  # Check the equality of the dates

    assert resampled_df.loc['2023-10-03', 'value'] == 2
    assert resampled_df.loc['2023-10-04', 'value'] == 3
    assert resampled_df.loc['2023-10-06', 'value'] == 3



def test_calculate_mae():
    true_values = pd.Series([1, 2, 3, 4, 5])
    predicted_values = pd.Series([1.1, 2.1, 2.9, 3.9, 5.1])
    mae = calculate_mae(true_values, predicted_values)
    assert mae == 0.1, "Mean Absolute Error should be 0.1"

def test_calculate_mae_with_missing_values():
    true_values = pd.Series([1, 2, 3, 4, 5])
    predicted_values = pd.Series([1.1, 2.1, 2.9, 3.9, None])
    mae = calculate_mae(true_values, predicted_values)
    assert mae == 0.1, "Mean Absolute Error should be 0.1"
    assert mae == calculate_mae(true_values, predicted_values, ignore_missing_values=False), "Mean Absolute Error should be 0.1"
    assert mae == calculate_mae(true_values, predicted_values, ignore_missing_values=True), "Mean Absolute Error should be 0.1"
    assert mae == calculate_mae(true_values, predicted_values, ignore_missing_values=False, ignore_nan_values=True), "Mean Absolute Error should be 0.1"
    assert mae == calculate_mae(true_values, predicted_values, ignore_missing_values=True, ignore_nan_values=True), "Mean Absolute Error should be 0.1"

def test_calculate_mae_with_nan_values():
    true_values = pd.Series([1, 2, 3, 4, 5])
    predicted_values = pd.Series([1.1, 2.1, 2.9, 3.9, np.nan])
    mae = calculate_mae(true_values, predicted_values)
    assert mae == 0.1, "Mean Absolute Error should be 0.1"
    assert mae == calculate_mae(true_values, predicted_values, ignore_missing_values=False), "Mean Absolute Error should be 0.1"
    assert mae == calculate_mae(true_values, predicted_values, ignore_missing_values=True), "Mean Absolute Error should be 0.1"
    assert mae == calculate_mae(true_values, predicted_values, ignore_missing_values=False, ignore_nan_values=True), "Mean Absolute Error should be 0.1"
    assert mae == calculate_mae(true_values, predicted_values, ignore_missing_values=True, ignore_nan_values=True), "Mean Absolute Error should be 0.1"

def test_calculate_mae_with_different_lengths():
    true_values = pd.Series([1, 2, 3, 4, 5])
    predicted_values = pd.Series([1.1, 2.1, 2.9, 3.9])
    mae = calculate_mae(true_values, predicted_values)
    assert mae == 0.1, "Mean Absolute Error should be 0.1"
    assert mae == calculate_mae(true_values, predicted_values, ignore_missing_values=False), "Mean Absolute Error should be 0.1"
    assert mae == calculate_mae(true_values, predicted_values, ignore_missing_values=True), "Mean Absolute Error should be 0.1"
    assert mae == calculate_mae(true_values, predicted_values, ignore_missing_values=False, ignore_nan_values=True), "Mean Absolute Error should be 0.1"
    assert mae == calculate_mae(true_values, predicted_values, ignore_missing_values=True, ignore_nan_values=True), "Mean Absolute Error should be 0.1"

def test_calculate_mae_with_different_lengths_and_nan_values():
    true_values = pd.Series([1, 2, 3, 4, 5])
    predicted_values = pd.Series([1.1, 2.1, 2.9, 3.9, np.nan])
    mae = calculate_mae(true_values, predicted_values)
    assert mae == 0.1, "Mean Absolute Error should be 0.1"
    assert mae == calculate_mae(true_values, predicted_values, ignore_missing_values=False), "Mean Absolute Error should be 0.1"
    assert mae == calculate_mae(true_values, predicted_values, ignore_missing_values=True), "Mean Absolute Error should be 0.1"
    assert mae == calculate_mae(true_values, predicted_values, ignore_missing_values=False, ignore_nan_values=True), "Mean Absolute Error should be 0.1"
    assert mae == calculate_mae(true_values, predicted_values, ignore_missing_values=True, ignore_nan_values=True), "Mean Absolute Error should be 0.1"