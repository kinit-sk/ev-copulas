"""Dataset preprocessing module for EV charging data.

This module provides functionality to preprocess EV charging datasets from different
sources (Trondheim, Dundee, and proprietary data) into a standardized format with
common columns: start_hour_shifted, duration, and energy.

The module also includes utilities for splitting datasets into train/validation/test sets.
"""

import polars as pl
from loguru import logger

from evcopulas.config import RAW_DATA_DIR

def split_dataframe(df, train_frac=0.8, val_frac=0.1, test_frac=0.1):
    """Split polars DataFrame into train, validation, and test sets.
    
    Splits the DataFrame sequentially (respecting order) based on the provided fractions.
    This is useful for time-series data where temporal order should be preserved.
    
    Args:
        df: Input polars DataFrame to split.
        train_frac: Fraction for training set. Defaults to 0.8.
        val_frac: Fraction for validation set. Defaults to 0.1.
        test_frac: Fraction for test set. Defaults to 0.1.
        
    Returns:
        Tuple of train, validation, and test DataFrames.
        
    Raises:
        AssertionError: If fractions don't sum to 1.0 (within 1e-6 tolerance).
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"
    
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    return train_df, val_df, test_df

def preprocess_trondheim_dataset():
    """Preprocess the Trondheim EV charging dataset.
    
    Loads and processes the Trondheim dataset with the following transformations:
    
    - Filters for private users only
    - Converts start time to hours from midnight
    - Shifts start hours by +24 if before 6 AM (for better modeling)
    - Removes null values in critical columns
    - Standardizes column names to: start_hour_shifted, duration, energy
    
    Returns:
        Processed DataFrame with columns:
        
            - start_hour_shifted: Start time in hours (6-30 range)
            - duration: Charging duration in hours
            - energy: Energy consumed in kWh
            
    Note:
        Expects raw data file at RAW_DATA_DIR / "trondheim.csv" with semicolon
        separator and decimal comma format.
    """
    logger.info("Preprocessing Trondheim dataset...")

    df = pl.read_csv(
        RAW_DATA_DIR / "trondheim.csv", 
        separator=";", 
        decimal_comma=True, 
        null_values="NA", 
        try_parse_dates=True
    ).filter(
        pl.col("User_type") == "Private",
    ).with_columns(
        start_datetime_hours = (pl.col("Start_plugin") - pl.col("Start_plugin").dt.date()).dt.total_seconds() / 60**2,
    ).drop_nulls(
        ["End_plugout", "Duration_hours", "El_kWh"]
    ).with_columns(
        start_datetime_hours_shifted = pl.when(pl.col("start_datetime_hours") < 6).then(pl.col("start_datetime_hours") + 24).otherwise(pl.col("start_datetime_hours")),
    ).select(
        "start_datetime_hours_shifted", "Duration_hours", "El_kWh"
    ).rename(
        {"start_datetime_hours_shifted": "start_hour_shifted", "Duration_hours": "duration", "El_kWh": "energy"}
    )

    logger.success(f"Processed data successfully.")

    return df

def preprocess_dundee_dataset():
    """Preprocess the Dundee EV charging dataset.
    
    Loads and processes the Dundee dataset with the following transformations:
    
    - Filters for rapid chargers only
    - Converts start time to hours from midnight
    - Shifts start hours by +24 if before 6 AM (for better modeling)
    - Removes null values in critical columns
    - Standardizes column names to: start_hour_shifted, duration, energy

    Returns:
        pl.DataFrame: Processed DataFrame with columns:

            - start_hour_shifted: Start time in hours (6-30 range)
            - duration: Charging duration in hours
            - energy: Energy consumed in kWh
    """
    logger.info("Preprocessing Dundee dataset...")

    df = pl.read_csv(
        RAW_DATA_DIR / "dundee.csv", 
        try_parse_dates=True
    ).drop(
        "Duration"
    ).filter(
        pl.col("Connector Type") == "rapid",
    ).with_columns(
        start_datetime_hours = (pl.col("Start") - pl.col("Start").dt.date()).dt.total_seconds() / 60**2,
    ).with_columns(
        Duration_hours = (pl.col("End") - pl.col("Start")).dt.total_seconds() / 60**2,
    ).filter(
        pl.col("Duration_hours").is_between(0,24)
    ).drop_nulls(
        ["Consum(kWh)"]
    ).with_columns(
        start_datetime_hours_shifted = pl.when(pl.col("start_datetime_hours") < 6).then(pl.col("start_datetime_hours") + 24).otherwise(pl.col("start_datetime_hours")),
    ).select(
        "start_datetime_hours_shifted", "Duration_hours", "Consum(kWh)"
    ).rename(
        {"start_datetime_hours_shifted": "start_hour_shifted", "Duration_hours": "duration", "Consum(kWh)": "energy"}
    )

    logger.success(f"Processed data successfully.")

    return df

def preprocess_proprietary_dataset():
    """
    Preprocess the proprietary EV charging dataset. This function shall be rewritten to 
    your own proprietary dataset if necessary.
    
    Returns:
        pl.DataFrame: Processed DataFrame with columns:

            - start_hour_shifted: Start time in hours (6-30 range)
            - duration: Charging duration in hours
            - energy: Energy consumed in kWh
    """
    logger.info("Preprocessing proprietary dataset...")

    df = pl.read_csv(
        RAW_DATA_DIR / "proprietary.csv",
    ).with_columns(
        pl.col("Start Time").str.to_datetime("%-m/%-d/%y %-H:%M"),
        pl.col("Stop Time").str.to_datetime("%-m/%-d/%y %-H:%M"),
    ).with_columns(
        start_datetime_hours = (pl.col("Start Time") - pl.col("Start Time").dt.date()).dt.total_seconds() / 60**2,
    ).with_columns(
        Duration_hours = pl.col("Duration (min)") / 60,
    ).with_columns(
        kWh = pl.col("Wh") / 1000
    ).sort(
        "Start Time"
    ).filter(
        pl.col("Duration_hours").is_between(0, 24)
    ).drop_nulls(
        ["kWh"]
    ).with_columns(
        start_datetime_hours_shifted = pl.when(pl.col("start_datetime_hours") < 6).then(pl.col("start_datetime_hours") + 24).otherwise(pl.col("start_datetime_hours")),
    ).select(
        "start_datetime_hours_shifted", "Duration_hours", "kWh"
    ).rename(
        {"start_datetime_hours_shifted": "start_hour_shifted", "Duration_hours": "duration", "kWh": "energy"}
    )

    logger.success(f"Processed data successfully.")

    return df

