import os
import pandas as pd
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_dataset(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(BASE_DIR, "data", "raw", "clickbait_data.csv")
    df = pd.read_csv(csv_path)

    print("Dataset loaded successfully.")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns)

    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nLabel distribution:")
    print(df["clickbait"].value_counts())

    return df


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_dataset(df):
    df = df[["headline", "clickbait"]].copy()

    df = df.dropna()
    df = df.drop_duplicates()

    df["cleaned_headline"] = df["headline"].apply(clean_text)

    print("\nAfter cleaning:")
    print(df.head())

    return df


if __name__ == "__main__":
    df = load_dataset()
    cleaned_df = clean_dataset(df)

    output_path = os.path.join(BASE_DIR, "data", "processed", "cleaned_clickbait_data.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)

    print(f"\nCleaned dataset saved to {output_path}")