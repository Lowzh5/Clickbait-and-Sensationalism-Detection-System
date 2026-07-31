import os #file path operations (joining paths, getting directory names)
import pandas as pd
import re #regular expressions
from sklearn.feature_extraction.text import TfidfVectorizer #implement TF-IDF 
from sklearn.model_selection import train_test_split # split the dataset to 80% training and 20% testing
import joblib # store the tfidf_vectorizer.pkl

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # find the project root(not strictly necessary but just to avoid headaches)

def load_dataset(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(BASE_DIR, "data", "raw", "clickbait_data.csv") # project root/data/raw/clickbait_data.csv
    df = pd.read_csv(csv_path) #read the CSV file into a pandas Dataframe

    print("Dataset loaded successfully.")
    print("Shape:", df.shape) #row , column count
    print("\nColumns:")
    print(df.columns) # column names

    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nMissing values:")
    print(df.isnull().sum()) #count missing values per column

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nLabel distribution:")
    print(df["clickbait"].value_counts()) # count the total number of clickbait/non-clickbait

    return df

def clean_text(text):
    text = str(text).lower() # convert to str then lowercases everything
    text = text.replace("'","") # for the contraction purpose " ' "
    # re.sub(pattern[for those not in this pattern], repl, string)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) # remove all special character, replace those with a space
    text = re.sub(r"\s+", " ", text).strip() # collapses multiple spaces into a single sapce, strip removes leading/trainling spaces
    return text

def clean_dataset(df):
    df = df[["headline", "clickbait"]].copy() #create a separate copy to avoid accidentally modify the original DataFrame

    #apply the clean text to each headline and store it to a new column call cleaned_headline
    df["cleaned_headline"] = df["headline"].apply(clean_text)

    print("\nAfter cleaning:")
    print(df.head())

    return df

def tfidf(df):
    X = df["cleaned_headline"]
    y = df["clickbait"]

    # seperate the train/test(80/20)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42) #random_state = ensure each time is using '42' style to split the data
    
    # build TF-IDF vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1,2)) # if the performance is quite slow then add the max_features
    X_train_tfidf = vectorizer.fit_transform(X_train) #fit_transfrom is learn + change
    X_test_tfidf = vectorizer.transform(X_test) # 从vectorizer里面拿X_train的词典来train

    # save vectorizer
    vectorizer_path = os.path.join(BASE_DIR,"models","tfidf_vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path) #save the model(vectorizer) to the file(vectorizer_path)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

"""
only run when execute this file directly, it does not run when another 
file import from it (from shared_pipeline import clean_text) 
"""
if __name__ == "__main__": 
    df = load_dataset()
    cleaned_df = clean_dataset(df)

    # save cleaned data
    output_path = os.path.join(BASE_DIR, "data", "processed", "cleaned_clickbait_data.csv")
    cleaned_df.to_csv(output_path, index=False) # saves the cleaned DataFrame to CSV. index=False means don't write the row numbers into the file

    #build TF-IDF
    X_train, X_test, y_train, y_test, vectorizer = tfidf(cleaned_df)
    print(f"\nTF-IDF done. Train shape: {X_train.shape}, Test shape: {X_test.shape}")