# ========================================
# SENTIMENT ANALYSIS WEB APP
# Complete NLP Application with Streamlit
# ========================================

import streamlit as st
import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from collections import Counter
import plotly.express as px

# ========================================
# DOWNLOAD NLTK RESOURCES
# ========================================

def download_nltk_resources():
    """Download required NLTK resources"""
    resources = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'vader_lexicon',
        'averaged_perceptron_tagger'
    ]
    
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

download_nltk_resources()

# ========================================
# PAGE CONFIGURATION
# ========================================

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CUSTOM CSS
# ========================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .positive-box {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .negative-box {
        background-color: #f8d7da;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    .neutral-box {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    .stButton button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #145a8d;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# HEADER
# ========================================

st.markdown('<p class="main-header">😊 Sentiment Analysis App</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyze the sentiment of any text using Natural Language Processing</p>', unsafe_allow_html=True)

# ========================================
# SIDEBAR - INPUT
# ========================================

st.sidebar.header("📝 Text Input")

# Text input
text_input = st.sidebar.text_area(
    "Enter your text:",
    height=200,
    value="I absolutely loved this movie! The acting was brilliant and the story was captivating. Highly recommended! ⭐⭐⭐⭐⭐"
)

# Sample texts
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Sample Texts")

if st.sidebar.button("🟢 Positive Review"):
    text_input = "This product is amazing! I love the quality and design. It's worth every penny. Highly recommend! ⭐⭐⭐⭐⭐"

if st.sidebar.button("🔴 Negative Review"):
    text_input = "Terrible product. Broke after 2 days. Customer service was useless. Complete waste of money. DO NOT BUY!"

if st.sidebar.button("🟡 Neutral Review"):
    text_input = "The product arrived on time. It works as expected. Nothing special but gets the job done."

if st.sidebar.button("😊 Tweets"):
    text_input = "Just tried the new AI tool! It's incredible! #MachineLearning #AI #NLP 🤖"

if st.sidebar.button("📰 News"):
    text_input = "The company announced record profits for Q4 2023. The CEO said 'We're very proud of our achievements.'"

# ========================================
# SENTIMENT ANALYSIS FUNCTIONS
# ========================================

def preprocess_text(text):
    """Clean and preprocess text"""
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_sentiment_vader(text):
    """Get sentiment using VADER"""
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    
    if scores['compound'] >= 0.05:
        sentiment = "Positive 😊"
    elif scores['compound'] <= -0.05:
        sentiment = "Negative 😞"
    else:
        sentiment = "Neutral 😐"
    
    return sentiment, scores

def get_sentiment_textblob(text):
    """Get sentiment using TextBlob"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    if polarity > 0.1:
        sentiment = "Positive 😊"
    elif polarity < -0.1:
        sentiment = "Negative 😞"
    else:
        sentiment = "Neutral 😐"
    
    return sentiment, polarity, subjectivity

def extract_keywords(text):
    """Extract keywords from text"""
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    keywords = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    # Get frequency
    freq = Counter(keywords)
    
    return freq.most_common(10)

def create_wordcloud(text):
    """Create word cloud"""
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(text)
    
    return wordcloud

# ========================================
# MAIN ANALYSIS
# ========================================

if st.sidebar.button("🔮 Analyze Sentiment", type="primary"):
    
    with st.spinner("Analyzing text..."):
        
        # Preprocess text
        cleaned_text = preprocess_text(text_input)
        
        # ========================================
        # SENTIMENT RESULTS
        # ========================================
        
        st.subheader("📊 Sentiment Analysis Results")
        
        # Get sentiment from both methods
        sentiment_vader, scores = get_sentiment_vader(text_input)
        sentiment_blob, polarity, subjectivity = get_sentiment_textblob(text_input)
        
        # Display main result
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if "Positive" in sentiment_vader:
                st.markdown('<div class="positive-box">', unsafe_allow_html=True)
                st.markdown(f"### {sentiment_vader}")
                st.markdown("✅ Positive sentiment detected!")
                st.markdown("</div>", unsafe_allow_html=True)
            elif "Negative" in sentiment_vader:
                st.markdown('<div class="negative-box">', unsafe_allow_html=True)
                st.markdown(f"### {sentiment_vader}")
                st.markdown("❌ Negative sentiment detected!")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="neutral-box">', unsafe_allow_html=True)
                st.markdown(f"### {sentiment_vader}")
                st.markdown("😐 Neutral sentiment detected!")
                st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.metric("VADER Sentiment", sentiment_vader)
            st.metric("TextBlob Polarity", f"{polarity:.2f}")
        
        with col3:
            st.metric("TextBlob Subjectivity", f"{subjectivity:.2f}")
            st.metric("Text Length", len(text_input))
        
        # ========================================
        # DETAILED SCORES
        # ========================================
        
        st.subheader("📈 Sentiment Scores")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**VADER Scores:**")
            scores_df = pd.DataFrame({
                'Metric': ['Positive', 'Negative', 'Neutral', 'Compound'],
                'Score': [scores['pos'], scores['neg'], scores['neu'], scores['compound']]
            })
            st.dataframe(scores_df)
        
        with col2:
            # Visualize scores
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ['green', 'red', 'gray', 'blue']
            ax.bar(scores_df['Metric'], scores_df['Score'], color=colors)
            ax.set_title('VADER Sentiment Scores')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # ========================================
        # TEXT ANALYSIS
        # ========================================
        
        st.subheader("📝 Text Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Text:**")
            st.info(text_input)
            
            st.write("**Cleaned Text:**")
            st.success(cleaned_text)
        
        with col2:
            # Word count
            words = text_input.split()
            st.metric("Word Count", len(words))
            st.metric("Character Count", len(text_input))
            st.metric("Sentence Count", len(text_input.split('.')))
        
        # ========================================
        # KEYWORD EXTRACTION
        # ========================================
        
        st.subheader("🔑 Top Keywords")
        
        keywords = extract_keywords(text_input)
        
        if keywords:
            col1, col2 = st.columns(2)
            
            with col1:
                keywords_df = pd.DataFrame(keywords, columns=['Word', 'Frequency'])
                st.dataframe(keywords_df)
            
            with col2:
                # Horizontal bar chart
                fig, ax = plt.subplots(figsize=(6, 4))
                words, freqs = zip(*keywords[:8])
                ax.barh(words, freqs, color='teal')
                ax.set_title('Top Keywords Frequency')
                ax.set_xlabel('Frequency')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        else:
            st.info("No keywords found")
        
        # ========================================
        # WORD CLOUD
        # ========================================
        
        st.subheader("☁️ Word Cloud")
        
        if len(cleaned_text) > 10:
            wordcloud = create_wordcloud(cleaned_text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("Text too short for word cloud")
        
        # ========================================
        # SENTIMENT BREAKDOWN
        # ========================================
        
        st.subheader("🎯 Sentiment Breakdown")
        
        # Create pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        values = [scores['pos'], scores['neu'], scores['neg']]
        colors = ['#28a745', '#ffc107', '#dc3545']
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Sentiment Distribution')
        st.pyplot(fig)
        
        # ========================================
        # SENTENCE-BY-SENTENCE ANALYSIS
        # ========================================
        
        st.subheader("📝 Sentence-by-Sentence Analysis")
        
        sentences = text_input.split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if sentences:
            sent_data = []
            for sent in sentences:
                _, scores = get_sentiment_vader(sent)
                sentiment = "Positive" if scores['compound'] >= 0.05 else "Negative" if scores['compound'] <= -0.05 else "Neutral"
                sent_data.append({
                    'Sentence': sent[:50] + '...' if len(sent) > 50 else sent,
                    'Sentiment': sentiment,
                    'Score': scores['compound']
                })
            
            df_sent = pd.DataFrame(sent_data)
            
            # Color code
            def color_sentiment(val):
                if val == 'Positive':
                    return 'background-color: #d4edda'
                elif val == 'Negative':
                    return 'background-color: #f8d7da'
                else:
                    return 'background-color: #fff3cd'
            
            st.dataframe(df_sent.style.applymap(color_sentiment, subset=['Sentiment']))
        else:
            st.info("No complete sentences found")

# ========================================
# EDUCATIONAL SECTION
# ========================================

st.sidebar.markdown("---")
with st.sidebar.expander("📚 How Sentiment Analysis Works"):
    st.markdown("""
    **VADER (Valence Aware Dictionary and sEntiment Reasoner)**
    
    - Rule-based sentiment analysis
    - Designed for social media text
    - Returns four scores:
      - Positive (pos)
      - Negative (neg)
      - Neutral (neu)
      - Compound (-1 to +1)
    
    **TextBlob**
    
    - Pre-trained sentiment model
    - Returns polarity (-1 to +1)
    - Returns subjectivity (0 to 1)
    """)

with st.sidebar.expander("💡 Sentiment Classification"):
    st.markdown("""
    **Positive 😊**
    - Compound score ≥ 0.05
    - Polarity > 0.1
    - Keywords: love, great, amazing
    
    **Negative 😞**
    - Compound score ≤ -0.05
    - Polarity < -0.1
    - Keywords: bad, terrible, hate
    
    **Neutral 😐**
    - Compound score between -0.05 and 0.05
    - Polarity between -0.1 and 0.1
    - Factual statements
    """)

# ========================================
# FOOTER
# ========================================

st.markdown("---")
st.caption("😊 Sentiment Analysis App v1.0 - Built with Streamlit, NLTK, and TextBlob")

# ========================================
# RUN THE APP
# ========================================

# To run: streamlit run sentiment_analysis_app.pystreamlit run "sentiment_analysis_app.py"