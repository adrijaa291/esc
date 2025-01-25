# SMS/Email Classifier

This is an SMS/Email classifier built with Streamlit and machine learning. It classifies input messages as either **Spam** or **Not Spam** using a trained model and text preprocessing techniques.

## Features
- **Text Preprocessing**: Cleans and preprocesses input text by removing stopwords, punctuation, and stemming words.
- **Spam Detection**: Classifies messages as spam or not using a trained machine learning model.
- **Streamlit Interface**: Simple web interface for inputting messages and displaying results.

## How It Works
1. **Input**: User enters an SMS or email message.
2. **Preprocessing**: The input text is tokenized, cleaned, and stemmed to prepare it for prediction.
3. **Vectorization**: The cleaned text is transformed into numerical features using a TF-IDF vectorizer.
4. **Prediction**: The model predicts whether the message is spam or not.

The model is trained using a dataset of SMS messages, and the text transformation process ensures high accuracy in classification.
