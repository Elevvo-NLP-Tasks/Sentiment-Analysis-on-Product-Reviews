import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import re

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def preprocess_text(text):
    if pd.isna(text):
        return ""
    
    # Lowercase
    text = str(text).lower()
    
    # Remove HTML tags (common in IMDB dataset)
    text = re.sub(r'<br\s*/?>', ' ', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove punctuation and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(lemmatized)

def load_and_preprocess_data(dataset_path):
    print("Loading data...")
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Map sentiment to binary (1 for positive, 0 for negative)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    print("Preprocessing text data... (this will take some time)")
    # To speed up testing, you might want to slice df (e.g. df = df.head(5000)) here
    df['Processed_Text'] = df['review'].apply(preprocess_text)
    
    print("Splitting into train and test sets...")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = load_and_preprocess_data('archive/IMDB_Dataset.csv')
    print("Data preprocessing completed successfully!")
    print(f"Training set shape: {train_df.shape}")
    print(f"Testing set shape: {test_df.shape}")
