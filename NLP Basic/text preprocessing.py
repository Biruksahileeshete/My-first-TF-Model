# ========================================
# NLP BASIC TEXT PREPROCESSING APP
# Complete Web Application with Streamlit
# ========================================

import streamlit as st
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

# Download required NLTK data (first time only)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

# ========================================
# PAGE CONFIGURATION
# ========================================

st.set_page_config(
    page_title="NLP Text Preprocessing",
    page_icon="📝",
    layout="wide"
)

# ========================================
# HEADER
# ========================================

st.title("📝 NLP Text Preprocessing App")
st.markdown("Clean, process, and analyze text data for Natural Language Processing!")

# ========================================
# SIDEBAR - INPUT
# ========================================

st.sidebar.header("📝 Input Text")
text_input = st.sidebar.text_area(
    "Enter your text:",
    height=200,
    value="Hello World! This is a sample text for NLP preprocessing. It contains numbers like 123 and special characters like @#$%. Let's clean it up! :)"
)

# ========================================
# SIDEBAR - PREPROCESSING OPTIONS
# ========================================

st.sidebar.header("⚙️ Preprocessing Options")

lowercase = st.sidebar.checkbox("Convert to Lowercase", value=True)
remove_punctuation = st.sidebar.checkbox("Remove Punctuation", value=True)
remove_numbers = st.sidebar.checkbox("Remove Numbers", value=True)
remove_stopwords = st.sidebar.checkbox("Remove Stopwords", value=True)
stemming = st.sidebar.checkbox("Apply Stemming", value=False)
lemmatization = st.sidebar.checkbox("Apply Lemmatization", value=False)
tokenization = st.sidebar.checkbox("Tokenize Words", value=True)

# ========================================
# TEXT PREPROCESSING FUNCTIONS
# ========================================

def preprocess_text(text, options):
    """Apply preprocessing steps based on options"""
    
    results = {
        'original': text,
        'steps': []
    }
    
    step_text = text
    
    # Step 1: Lowercase
    if options['lowercase']:
        step_text = step_text.lower()
        results['steps'].append({'name': 'Lowercase', 'text': step_text})
    
    # Step 2: Remove punctuation
    if options['remove_punctuation']:
        step_text = re.sub(r'[^\w\s]', ' ', step_text)
        step_text = re.sub(r'\s+', ' ', step_text).strip()
        results['steps'].append({'name': 'Remove Punctuation', 'text': step_text})
    
    # Step 3: Remove numbers
    if options['remove_numbers']:
        step_text = re.sub(r'\d+', '', step_text)
        step_text = re.sub(r'\s+', ' ', step_text).strip()
        results['steps'].append({'name': 'Remove Numbers', 'text': step_text})
    
    # Step 4: Tokenization
    if options['tokenization']:
        tokens = word_tokenize(step_text)
        results['tokens'] = tokens
        results['steps'].append({'name': 'Tokenization', 'text': str(tokens)})
    
    # Step 5: Remove stopwords
    if options['remove_stopwords'] and 'tokens' in results:
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in results['tokens'] if word.lower() not in stop_words]
        results['tokens_no_stopwords'] = filtered_tokens
        results['steps'].append({'name': 'Remove Stopwords', 'text': str(filtered_tokens)})
    
    # Step 6: Stemming
    if options['stemming'] and 'tokens' in results:
        stemmer = PorterStemmer()
        if 'tokens_no_stopwords' in results:
            stemmed_tokens = [stemmer.stem(word) for word in results['tokens_no_stopwords']]
        else:
            stemmed_tokens = [stemmer.stem(word) for word in results['tokens']]
        results['stemmed_tokens'] = stemmed_tokens
        results['steps'].append({'name': 'Stemming', 'text': str(stemmed_tokens)})
    
    # Step 7: Lemmatization
    if options['lemmatization'] and 'tokens' in results:
        lemmatizer = WordNetLemmatizer()
        if 'tokens_no_stopwords' in results:
            lemmatized_tokens = [lemmatizer.lemmatize(word) for word in results['tokens_no_stopwords']]
        else:
            lemmatized_tokens = [lemmatizer.lemmatize(word) for word in results['tokens']]
        results['lemmatized_tokens'] = lemmatized_tokens
        results['steps'].append({'name': 'Lemmatization', 'text': str(lemmatized_tokens)})
    
    # Final cleaned text
    if 'tokens_no_stopwords' in results:
        results['cleaned_text'] = ' '.join(results['tokens_no_stopwords'])
    elif 'tokens' in results:
        results['cleaned_text'] = ' '.join(results['tokens'])
    else:
        results['cleaned_text'] = step_text
    
    return results

def get_text_stats(text):
    """Get statistics about the text"""
    stats = {
        'characters': len(text),
        'words': len(text.split()),
        'sentences': len(sent_tokenize(text)),
        'unique_words': len(set(text.split())),
        'avg_word_length': sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0
    }
    return stats

# ========================================
# MAIN CONTENT
# ========================================

# Process text when button is clicked
if st.sidebar.button("🔮 Process Text", type="primary"):
    
    # Prepare options
    options = {
        'lowercase': lowercase,
        'remove_punctuation': remove_punctuation,
        'remove_numbers': remove_numbers,
        'remove_stopwords': remove_stopwords,
        'stemming': stemming,
        'lemmatization': lemmatization,
        'tokenization': tokenization
    }
    
    # Process text
    results = preprocess_text(text_input, options)
    
    # ========================================
    # DISPLAY RESULTS
    # ========================================
    
    st.subheader("📊 Preprocessing Results")
    
    # Original text stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Characters", len(text_input))
    with col2:
        st.metric("Words", len(text_input.split()))
    with col3:
        st.metric("Sentences", len(sent_tokenize(text_input)))
    with col4:
        st.metric("Unique Words", len(set(text_input.split())))
    
    # Display preprocessing steps
    st.subheader("🔍 Preprocessing Steps")
    for step in results['steps']:
        with st.expander(f"📌 {step['name']}"):
            st.code(step['text'], language='text')
    
    # Display cleaned text
    st.subheader("✅ Cleaned Text")
    if 'cleaned_text' in results:
        st.success(results['cleaned_text'])
    
    # ========================================
    # TOKENIZATION RESULTS
    # ========================================
    
    if 'tokens' in results:
        st.subheader("📝 Tokenization Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Tokens:**")
            st.write(results['tokens'])
        
        if 'tokens_no_stopwords' in results:
            with col2:
                st.markdown("**Tokens without Stopwords:**")
                st.write(results['tokens_no_stopwords'])
    
    # ========================================
    # STEMMING VS LEMMATIZATION COMPARISON
    # ========================================
    
    if 'stemmed_tokens' in results and 'lemmatized_tokens' in results:
        st.subheader("🔄 Stemming vs Lemmatization Comparison")
        
        comparison_data = {
            'Original': results['tokens'][:10] if 'tokens' in results else [],
            'Stemmed': results['stemmed_tokens'][:10],
            'Lemmatized': results['lemmatized_tokens'][:10]
        }
        
        df_compare = pd.DataFrame(comparison_data)
        st.dataframe(df_compare)
        
        st.info("""
        **Stemming vs Lemmatization:**
        - **Stemming**: Cuts word endings (e.g., "running" → "run")
        - **Lemmatization**: Reduces to dictionary form (e.g., "running" → "run")
        - Lemmatization is more accurate but slower
        """)
    
    # ========================================
    # WORD CLOUD
    # ========================================
    
    if 'tokens' in results:
        st.subheader("☁️ Word Cloud")
        
        # Choose tokens for word cloud
        word_list = results.get('tokens_no_stopwords', results.get('tokens', []))
        
        if word_list:
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis'
            ).generate(' '.join(word_list))
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("No words available for word cloud")
    
    # ========================================
    # FREQUENCY ANALYSIS
    # ========================================
    
    if 'tokens' in results:
        st.subheader("📊 Word Frequency Analysis")
        
        # Get frequency distribution
        word_list = results.get('tokens_no_stopwords', results.get('tokens', []))
        word_freq = Counter(word_list)
        
        # Top 10 words
        top_words = word_freq.most_common(10)
        
        if top_words:
            fig, ax = plt.subplots(figsize=(10, 5))
            words, counts = zip(*top_words)
            ax.bar(words, counts, color='skyblue', edgecolor='black')
            ax.set_title('Top 10 Most Frequent Words')
            ax.set_xlabel('Words')
            ax.set_ylabel('Frequency')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Show as table
            df_freq = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
            st.dataframe(df_freq)
        else:
            st.warning("No words to analyze")
    
    # ========================================
    # POS TAGGING
    # ========================================
    
    if 'tokens' in results:
        st.subheader("📚 Part-of-Speech (POS) Tagging")
        
        if st.checkbox("Show POS Tags"):
            try:
                tokens_to_tag = results.get('tokens_no_stopwords', results.get('tokens', []))
                pos_tags = pos_tag(tokens_to_tag)
                
                # Create DataFrame
                pos_df = pd.DataFrame(pos_tags, columns=['Word', 'POS Tag'])
                st.dataframe(pos_df)
                
                # POS distribution
                pos_counts = Counter([tag for _, tag in pos_tags])
                pos_df_counts = pd.DataFrame(pos_counts.items(), columns=['POS Tag', 'Count'])
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(pos_df_counts['POS Tag'], pos_df_counts['Count'], color='lightgreen', edgecolor='black')
                ax.set_title('POS Tag Distribution')
                ax.set_xlabel('POS Tag')
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                st.info("""
                **Common POS Tags:**
                - **NN**: Noun
                - **VB**: Verb
                - **JJ**: Adjective
                - **RB**: Adverb
                - **DT**: Determiner
                - **PRP**: Pronoun
                """)
            except Exception as e:
                st.error(f"POS tagging error: {e}")

# ========================================
# SAMPLE TEXTS
# ========================================

with st.sidebar.expander("📝 Sample Texts"):
    if st.button("Sample 1: Product Review"):
        st.session_state.text_input = "The product is amazing! I love the quality and design. It's worth every penny. Highly recommend! ⭐⭐⭐⭐⭐"
    
    if st.button("Sample 2: Tweets"):
        st.session_state.text_input = "Just tried the new #AI tool! It's incredible! #MachineLearning #NLP #DataScience 🤖💻"
    
    if st.button("Sample 3: News Article"):
        st.session_state.text_input = "The company announced record profits for Q4 2023. The CEO said 'We're very proud of our achievements.'"
    
    if st.button("Sample 4: Mixed Text"):
        st.session_state.text_input = "Hello! This is a test. It contains numbers 123, symbols @#$%, and UPPERCASE words. Let's process this text! 😊"

# ========================================
# EDUCATIONAL SECTION
# ========================================

st.sidebar.markdown("---")
with st.sidebar.expander("📚 What is Text Preprocessing?"):
    st.markdown("""
    **Text preprocessing** is cleaning and preparing text for analysis:
    
    1. **Lowercase**: Makes text uniform
    2. **Remove Punctuation**: Removes . , ! ? etc.
    3. **Remove Numbers**: Removes digits
    4. **Tokenization**: Split into words
    5. **Stopwords**: Remove common words
    6. **Stemming**: Reduce to root form
    7. **Lemmatization**: Reduce to dictionary form
    """)

# ========================================
# FOOTER
# ========================================

st.markdown("---")
st.caption("📝 NLP Text Preprocessing App v1.0 - Learn Natural Language Processing")

# ========================================
# CUSTOM CSS
# ========================================

st.markdown("""
<style>
    .stButton button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .st-expander {
        background-color: #f0f2f6;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)