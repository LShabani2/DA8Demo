# utils.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


"""
    Filters and prepares the dataset for binary classification.
    Selects only the top 2 genres and encodes them using one-hot encoding.
    Also creates the binary target variable 'high_rating'.

    Parameters:
        df (pd.DataFrame): Original movie dataset.
        top2_genres (list): List of two genres to include.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) — Train/test splits.
"""
def prepare_top2_genre_data(df, top2_genres=['Drama', 'Comedy']):
    df = df.copy()
    df['high_rating'] = df['Rating'].apply(lambda x: 1 if x >= 4 else 0)
    df['GenreList'] = df['Genre'].dropna().apply(lambda x: [g.strip() for g in x.split(',')])
    df = df.explode('GenreList')
    df = df[df['GenreList'].isin(top2_genres)]

    X = pd.get_dummies(df['GenreList']) 
    y = df['high_rating']

    return train_test_split(X, y, test_size=0.2, random_state=42)

"""
    Trains and evaluates a k-Nearest Neighbors classifier.

    Parameters:
        X_train, X_test: Training and test features.
        y_train, y_test: Training and test labels.
        k (int): Number of neighbors to use.

    Returns:
        dict: Model name, accuracy, classification report, and confusion matrix.
"""
def run_knn(X_train, X_test, y_train, y_test, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return {
        'name': f'kNN (k={k})',
        'accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred, output_dict=False),
        'confusion': confusion_matrix(y_test, y_pred)
    }

"""
    Trains and evaluates a Decision Tree classifier.

    Parameters:
        X_train, X_test: Training and test features.
        y_train, y_test: Training and test labels.

    Returns:
        dict: Model name, accuracy, classification report, and confusion matrix.
"""
def run_decision_tree(X_train, X_test, y_train, y_test):
    dtree = DecisionTreeClassifier(random_state=42)
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)
    return {
        'name': 'Decision Tree',
        'accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred, output_dict=False),
        'confusion': confusion_matrix(y_test, y_pred)
    }

"""
    Nicely formats and prints key evaluation metrics for a classification model.

    Parameters:
        result (dict): Output dictionary from either run_knn or run_decision_tree.
"""
def print_model_results(result):
    print(f"\nModel: {result['name']}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print("Classification Report:")
    print(result['report'])
    print("Confusion Matrix:")
    print(result['confusion'])

# kNN & Decision Tree Results Visualization

