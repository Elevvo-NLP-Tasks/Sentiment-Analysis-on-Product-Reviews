import argparse
from data_preprocess import load_and_preprocess_data
from train import train_and_evaluate
from plots import plot_frequent_words
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train Sentiment Analysis Model")
    parser.add_argument('--data_path', type=str, default='archive/IMDB_Dataset.csv', help='Path to dataset')
    parser.add_argument('--max_words', type=int, default=10000, help='Maximum number of words for TF-IDF')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load and Preprocess Data
    train_df, test_df = load_and_preprocess_data(args.data_path)
    
    # 2. Visualize frequent words before training
    # Generating word clouds for the training data
    plot_frequent_words(train_df, text_col='Processed_Text', label_col='label')
    
    # 3. Vectorize text using TF-IDF
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=args.max_words)
    
    # Fit on training data and transform both training and testing data
    X_train = vectorizer.fit_transform(train_df['Processed_Text'])
    X_test = vectorizer.transform(test_df['Processed_Text'])
    
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    print(f"TF-IDF X_train shape: {X_train.shape}")
    print(f"TF-IDF X_test shape: {X_test.shape}")
    
    # Save the vectorizer
    os.makedirs('models', exist_ok=True)
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    print("Vectorizer saved to models/tfidf_vectorizer.pkl")
    
    # 4. Train Model and compare LR vs Naive Bayes
    best_model = train_and_evaluate(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
