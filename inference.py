import joblib
import pandas as pd
from data_preprocess import preprocess_text

def predict(text, model, vectorizer):
    """
    Preprocesses the text, vectorizes it, and predicts the sentiment.
    """
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]
    
    return "Positive" if prediction == 1 else "Negative"

def main():
    print("Loading model and vectorizer...")
    try:
        model = joblib.load('models/sentiment_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Please ensure the main pipeline is executed and models are saved.")
        return

    print("Sampling 5 random rows from the dataset...")
    # Load dataset to get some sample data
    try:
        df = pd.read_csv('archive/IMDB_Dataset.csv')
        sample_df = df.sample(n=5, random_state=42)
    except FileNotFoundError:
        print("Dataset not found. Using hardcoded examples instead.")
        sample_df = pd.DataFrame({
            'review': [
                "This movie was fantastic! I absolutely loved it.",
                "Terrible acting and a boring plot. Do not recommend.",
                "It was okay, not the best but not the worst.",
                "A brilliant masterpiece of cinema.",
                "I fell asleep halfway through. Total waste of time."
            ],
            'sentiment': ['positive', 'negative', 'negative', 'positive', 'negative']
        })
    
    print("\n" + "="*80)
    for index, row in sample_df.iterrows():
        review = str(row['review']) if pd.notna(row['review']) else ""
        actual_sentiment = str(row.get('sentiment', 'unknown')).capitalize()
        
        # Get prediction
        predicted_sentiment = predict(review, model, vectorizer)
        
        # Print input alongside prediction (truncate if too long)
        display_review = review[:250] + "..." if len(review) > 250 else review
        print(f"INPUT TEXT: {display_review}")
        print(f"=> ACTUAL: {actual_sentiment} | PREDICTION: {predicted_sentiment}")
        print("-" * 80)

if __name__ == "__main__":
    main()
