import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def prepare_data():

    df = pd.read_csv("MScCardsDatasetNew.csv")
    print(df.head())
    df = df[df["league"] == "EngPr"].copy()
    # Target variables
    target_vars = ["team1_yc", "team2_yc", "team1_rc", "team2_rc"]

    # Feature variables
    feature_vars = [
        "season",
        "kick_off_datetime",
        "team1_name", "team2_name",
        "referee",
        "attendance_value",
        "limited_audience",
        "stadium_surface",
        "stadium_capacity",
        "stadium_altitude",
        "distance",
        "team1_stadium_dist", "team2_stadium_dist"
    ]

    # Final selection
    chosen_vars = target_vars + feature_vars
    df = df[chosen_vars]
    list(df)
    df["kick_off_datetime"] = pd.to_datetime(df["kick_off_datetime"], errors="coerce")
    df["kickoff_year"] = df["kick_off_datetime"].dt.year
    df["kickoff_month"] = df["kick_off_datetime"].dt.month
    df["kickoff_day"] = df["kick_off_datetime"].dt.day
    df["kickoff_dayOfWeek"] = df["kick_off_datetime"].dt.dayofweek
    df["kickoff_hour"] = df["kick_off_datetime"].dt.hour

    df.drop(columns=["kick_off_datetime"], inplace=True)

    list(df)

    if "limited_audience" in df.columns:
        df["limited_audience"] = (
            df["limited_audience"]
            .astype(str).str.strip().str.upper()   
            .map({"TRUE": 1})                      
            .fillna(0)                             
            .astype(int)                           
        )
        
    y = df[target_vars].copy()
    X = df.drop(columns=target_vars).copy()

    # explicit column types for preprocessing
    numeric_features = [
        "attendance_value", 
        "stadium_capacity", 
        "stadium_altitude",
        "distance", 
        "team1_stadium_dist", 
        "team2_stadium_dist", 
        "limited_audience",               
        "kickoff_year", 
        "kickoff_month", 
        "kickoff_day", 
        "kickoff_dayOfWeek", 
        "kickoff_hour"
    ]
    categorical_features = [
        "season", 
        "team1_name", 
        "team2_name", 
        "referee", 
        "stadium_surface"
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")), 
        ("scaler", StandardScaler(with_mean=True, with_std=True)) 
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")), 
        ("onehot", OneHotEncoder(handle_unknown="ignore")) 
    ])

    final_numeric_features = []
    for col in numeric_features:
        if col in X.columns:
            final_numeric_features.append(col)

    final_categorical_features = []
    for col in categorical_features:
        if col in X.columns:
            final_categorical_features.append(col)
            
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, final_numeric_features),
            ("cat", categorical_transformer, final_categorical_features),
        ]
    )
    return X, y, preprocessor