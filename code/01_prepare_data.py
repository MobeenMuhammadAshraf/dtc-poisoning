"""
Prepare the DTC Recurrence dataset for poisoning experiments.


Input:  data/Thyroid_Diff.csv  (original UCI file)
Output: data/X_prepared_dedup.csv, data/y_prepared_dedup.csv
"""
import os
import pandas as pd


INPUT = "data/Thyroid_Diff.csv"
OUT_X = "data/X_prepared_dedup.csv"
OUT_Y = "data/y_prepared_dedup.csv"




def main():
    df = pd.read_csv(INPUT)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Positives (Recurred=Yes): {(df['Recurred'] == 'Yes').sum()}")


    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {n_before - len(df)} duplicate rows -> {len(df)} rows")
    print(f"Positives after dedup: {(df['Recurred'] == 'Yes').sum()}")


    y = (df["Recurred"] == "Yes").astype(int)
    X = pd.get_dummies(df.drop(columns=["Recurred", "Response"]), drop_first=False)
    print(f"Feature matrix: {X.shape}")


    X.to_csv(OUT_X, index=False)
    y.to_csv(OUT_Y, index=False)
    print(f"Saved {OUT_X} and {OUT_Y}")




if __name__ == "__main__":
    main()