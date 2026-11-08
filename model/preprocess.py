import numpy as np
import pandas as pd

# The three delinquency columns used by apply_fixed_cap and add_features
delinq_cols = [
    'NumberOfTimes90DaysLate',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTime60-89DaysPastDueNotWorse',
]


def apply_caps(df, bounds):
    """Clip columns to the IQR bounds learned from the training set."""
    df = df.copy()
    for col, (lower, upper) in bounds.items():
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def apply_fixed_cap(df):
    """Domain-knowledge caps."""
    df = df.copy()
    df['age'] = df['age'].clip(lower=18, upper=100)
    df[delinq_cols] = df[delinq_cols].clip(upper=10)
    df['NumberOfDependents'] = df['NumberOfDependents'].clip(upper=10)
    return df


def add_features(df):
    """Build the engineered columns the model was trained on."""
    df = df.copy()
    df['DelinquencyScore'] = df[delinq_cols].sum(axis=1)
    df['TotalLoans'] = df['NumberOfOpenCreditLinesAndLoans'] + df['NumberRealEstateLoansOrLines']
    return df


def preprocess(df, fill_values, bounds, fill_cols, log_cols, drop_cols):
    """Run raw applicant data through every step used in training."""
    df = df.copy()
    df[fill_cols] = df[fill_cols].fillna(fill_values)   # 1. fill missing
    df = apply_caps(df, bounds)                         # 2. IQR caps
    df = apply_fixed_cap(df)                            # 3. fixed caps
    df = add_features(df)                               # 4. new features
    df[log_cols] = np.log1p(df[log_cols])               # 5. log transform
    df = df.drop(columns=drop_cols)                     # 6. drop redundant columns
    return df
