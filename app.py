import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
from nltk.stem import SnowballStemmer
nltk.download('stopwords')
nltk.download('punkt')


def get_reviews(url):
    user_agent = ({'User-Agent':
			'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
			AppleWebKit/537.36 (KHTML, like Gecko) \
			Chrome/90.0.4430.212 Safari/537.36',
			'Accept-Language': 'en-US, en;q=0.5'})
    page = requests.get(url, headers = user_agent)
    soup = BeautifulSoup(page.text, 'html.parser')

    reviews = []
    for review in soup.find_all('div', class_='review-container'):
        rating = review.find('span', class_='ui_bubble_rating')['class'][1].split('_')[-1]
        #title = review.find('div', class_='quote').text.strip()
        content = review.find('div', class_='entry').find('p').text.strip()
        #date = review.find('span', class_='ratingDate')['title']
        reviews.append({
            'rating': rating,
            #'title': title,
            'content': content,
            #'date': date
        })
    next_page_link = soup.find('a', class_='nav next ui_button primary')
    if next_page_link:
        next_page_url = 'https://www.tripadvisor.com' + next_page_link['href']
        # Recursively call the function to get reviews from the next page
        reviews += get_reviews(next_page_url)    
    return reviews

def create_dataframe(data):
    df = pd.DataFrame(data, columns=['rating', 'content'])
    df['rating'] = df['rating'].astype(int) / 10
    return df.rename(columns={'content': 'review'})

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]|[\d]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)
    # Join the tokens back into a string
    text = ' '.join(tokens)

    return text

def sentiment_analysis(texts):
    # Load pre-trained sentiment analysis pipeline
    classifier = pipeline('sentiment-analysis')
    
    # Run sentiment analysis on input texts
    results = classifier(texts)
    
    return results



# Add a title to the app
st.title("Restaurant Reviews Sentiment Analysis")

# Get the restaurant URL from the user
url = st.text_input("Enter the TripAdvisor URL for the restaurant:")

# Add a button to submit the URL
if st.button("Submit"):
    # Get the reviews for the restaurant from TripAdvisor
    reviews = get_reviews(url)
    
    # Create a DataFrame from the reviews
    reviews_df = create_dataframe(reviews)
    
    # Preprocess the reviews
    reviews_df['review'] = reviews_df['review'].apply(preprocess_text)
    
    # Get the sentiment predictions for the reviews
    sentiments = sentiment_analysis(reviews_df['review'].tolist())
    
    # Add the sentiment predictions to the DataFrame
    reviews_df['sentiment'] = [result['label'] for result in sentiments]
    reviews_df['confidence'] = [result['score'] for result in sentiments]
    
    # Display the sentiment analysis summary
    st.write("Sentiment Analysis Summary:")
    st.write(reviews_df['sentiment'].value_counts())
    
    # Display a pie chart of the sentiment analysis results
    st.write("Sentiment Analysis Results:")
    fig, ax = plt.subplots()
    summary = reviews_df['sentiment'].value_counts()
    ax.pie(summary, labels=summary.index, autopct='%1.1f%%', startangle=90, colors=['#008080', '#E9967A'])
    ax.legend(title="Sentiment", loc="center right", bbox_to_anchor=(1, 0, 0.5, 1))
    st.pyplot(fig)
