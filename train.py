from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Initializes and trains both Logistic Regression and Naive Bayes models.
    Compares their accuracy and saves the better performing model.
    """
    print("\n--- Training Logistic Regression Model ---")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    lr_train_acc = lr_model.score(X_train, y_train)
    lr_test_acc = lr_model.score(X_test, y_test)
    print(f"Logistic Regression - Training Accuracy: {lr_train_acc:.4f}")
    print(f"Logistic Regression - Validation Accuracy: {lr_test_acc:.4f}")
    
    print("\n--- Training Naive Bayes Model ---")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    
    nb_train_acc = nb_model.score(X_train, y_train)
    nb_test_acc = nb_model.score(X_test, y_test)
    print(f"Naive Bayes - Training Accuracy: {nb_train_acc:.4f}")
    print(f"Naive Bayes - Validation Accuracy: {nb_test_acc:.4f}")
    
    # Determine the best model based on validation accuracy
    if lr_test_acc >= nb_test_acc:
        best_model = lr_model
        best_model_name = "Logistic Regression"
    else:
        best_model = nb_model
        best_model_name = "Naive Bayes"
        
    print(f"\nBest Model: {best_model_name}")
    
    # Save the best model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/sentiment_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"Saved the best model to {model_path}")
    
    return best_model
