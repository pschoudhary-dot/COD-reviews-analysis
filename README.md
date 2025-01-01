
## Files

- `call_of_duty_reviews_50000.csv`: The dataset containing 50,000 reviews of the Call of Duty game.
- `logistic_regression_model.pkl`: The trained Logistic Regression model.
- `naive_bayes_model.pkl`: The trained Naive Bayes model.
- `tfidf_vectorizer.pkl`: The trained TF-IDF vectorizer.
- `Sklrean_sentiment_analysisis_using_Naive_byes_and_logistic_regression.ipynb`: Jupyter notebook for training and evaluating the sentiment analysis models.
- `Text_analysis.ipynb`: Jupyter notebook for additional text analysis and visualization.

## Requirements

- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn
- tqdm
- langdetect
- emoji

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd cod_data
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Download NLTK data:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    ```

## Usage

1. Open the Jupyter notebooks:
    ```sh
    jupyter notebook
    ```

2. Run the cells in [Sklrean_sentiment_analysisis_using_Naive_byes_and_logistic_regression.ipynb](http://_vscodecontentref_/8) to train and evaluate the sentiment analysis models.

3. Run the cells in [Text_analysis.ipynb](http://_vscodecontentref_/9) for additional text analysis and visualization.

## Example

```python
# Load models and vectorizer
with open('naive_bayes_model.pkl', 'rb') as f:
    loaded_nb_model = pickle.load(f)

with open('logistic_regression_model.pkl', 'rb') as f:
    loaded_lr_model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

# Example usage of the loaded models
new_review = ["I used to love this game It's not very fantastic! don't ever try this game very bad"]

# Preprocess and transform the new review
new_review_tfidf = loaded_vectorizer.transform([preprocess_text(new_review[0])])

# Predict sentiment using Naive Bayes
predicted_sentiment_nb = loaded_nb_model.predict(new_review_tfidf)
print(f"Predicted Sentiment (Naive Bayes): {predicted_sentiment_nb[0]}")

# Predict sentiment using Logistic Regression
predicted_sentiment_lr = loaded_lr_model.predict(new_review_tfidf)
print(f"Predicted Sentiment (Logistic Regression): {predicted_sentiment_lr[0]}")
