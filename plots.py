import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

def plot_frequent_words(df, text_col='Processed_Text', label_col='label'):
    """
    Generates and saves a word cloud for positive and negative reviews.
    """
    print("Generating word clouds...")
    os.makedirs('plots', exist_ok=True)
    
    sentiments = {
        1: "Positive",
        0: "Negative"
    }
    
    for val, name in sentiments.items():
        # Get all text for the current sentiment
        text_data = " ".join(df[df[label_col] == val][text_col].dropna().values)
        
        if not text_data.strip():
            continue
            
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            max_words=150,
            colormap='viridis' if val == 1 else 'magma'
        ).generate(text_data)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        plt.title(f'Most Frequent Words - {name} Reviews', fontsize=16)
        
        save_path = f'plots/wordcloud_{name.lower()}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved word cloud for {name} to {save_path}")
