"""
FAQ Chatbot for CodeAlpha Internship Assignment
=================================================
A production-ready conversational AI chatbot using NLP techniques:
- Text preprocessing with NLTK (tokenization, stopwords removal, cleaning)
- TF-IDF vectorization for feature extraction
- Cosine similarity for intelligent FAQ matching
- Streamlit for professional chat UI with conversation memory
"""

import re
import nltk
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# NLTK DATA DOWNLOAD (Safe initialization with error handling)
# =============================================================================
@st.cache_resource
def download_nltk_data():
    """Download required NLTK datasets with proper error handling."""
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        pass
    
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass
    
    try:
        nltk.download('stopwords', quiet=True)
    except:
        pass

download_nltk_data()

# =============================================================================
# SIMPLE TOKENIZER (Fallback)
# =============================================================================
def simple_tokenize(text):
    """Simple tokenizer as fallback when NLTK fails."""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    tokens = text.split()
    return tokens

# =============================================================================
# EXPANDED FAQ DATASET (Added more refund-related questions)
# =============================================================================
FAQ_DATA = [
    {
        "question": "What is your return policy?",
        "answer": "We offer a 30-day return policy for all unused products in original packaging. Return shipping is free for defective items.",
        "keywords": ["return", "policy", "send back", "exchange", "refund item"]
    },
    {
        "question": "How long does shipping take?",
        "answer": "Standard shipping takes 3-5 business days. Express shipping (1-2 business days) is available at checkout for an additional fee.",
        "keywords": ["shipping", "delivery", "ship", "arrive", "shipping time"]
    },
    {
        "question": "Do you offer international shipping?",
        "answer": "Yes, we ship to over 50 countries worldwide. International delivery typically takes 7-14 business days depending on customs clearance.",
        "keywords": ["international", "overseas", "worldwide", "foreign", "abroad"]
    },
    {
        "question": "How can I track my order?",
        "answer": "Once your order ships, you'll receive a tracking number via email. You can also track your order from your account dashboard under 'Order History'.",
        "keywords": ["track", "tracking", "where is my order", "order status", "shipment"]
    },
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept all major credit cards (Visa, MasterCard, American Express), PayPal, Apple Pay, and Google Pay.",
        "keywords": ["payment", "pay", "credit card", "debit", "paypal"]
    },
    {
        "question": "Can I cancel or modify my order?",
        "answer": "Orders can be cancelled within 1 hour of placement. Modifications are not possible after processing begins, but you can return the item for a full refund.",
        "keywords": ["cancel", "modify", "change order", "update order", "order change"]
    },
    {
        "question": "How long does it take to get a refund?",
        "answer": "Refunds are processed within 5-7 business days after we receive your return. The money will be credited back to your original payment method. You'll receive an email confirmation once the refund is initiated.",
        "keywords": ["refund", "money back", "reimbursement", "refund time", "when refund", "get refund" ,"not receivede fund"]
    },
    {
        "question": "What is your refund process?",
        "answer": "To request a refund, first return the item within 30 days of purchase. Once we receive and inspect the return, we'll process your refund within 5-7 business days. You'll be notified via email at each step.",
        "keywords": ["refund process", "how to refund", "refund procedure", "return refund"]
    }
]

# =============================================================================
# ENHANCED TEXT PREPROCESSING (Preserves important words)
# =============================================================================
def preprocess_text(text: str, preserve_refund_keywords: bool = True) -> str:
    """
    Clean and normalize text for NLP processing with smarter filtering.
    
    Args:
        text: Raw user input or FAQ question string
        preserve_refund_keywords: Keep refund-related words even if they're stopwords
        
    Returns:
        Cleaned, normalized string ready for vectorization
    """
    # Step 1: Lowercase conversion
    text = text.lower()
    
    # Step 2: Remove special characters but keep important punctuation context
    text = re.sub(r'[^\w\s\?]', '', text)
    
    # Step 3: Tokenization with fallback
    try:
        tokens = word_tokenize(text)
    except (LookupError, Exception):
        tokens = simple_tokenize(text)
    
    # Important keywords that shouldn't be removed even if they're stopwords
    important_keywords = {'refund', 'return', 'money', 'when', 'get', 'how', 'long', 'track', 
                         'shipping', 'payment', 'cancel', 'order', 'delivery', 'international'}
    
    # Step 4: Remove stopwords with exception for important keywords
    try:
        stop_words = set(stopwords.words('english'))
        # Remove common stopwords but keep important ones
        tokens = [token for token in tokens 
                 if token not in stop_words or token in important_keywords]
    except (LookupError, Exception):
        # Basic stopwords list
        basic_stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'you', 'your', 
                          'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 
                          'a', 'an', 'and', 'if', 'or', 'because', 'as', 'until', 'while', 
                          'of', 'at', 'by', 'for', 'with', 'without', 'after', 'upon', 
                          'but', 'not', 'to', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 
                          'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 
                          'doing', 'the', 'so', 'yet'}
        tokens = [token for token in tokens if token not in basic_stopwords or token in important_keywords]
    
    # Step 5: Rejoin tokens
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text if cleaned_text else text  # Return original if cleaning removed everything

# =============================================================================
# ENHANCED SIMILARITY WITH KEYWORD BOOSTING
# =============================================================================
def get_best_response(user_question: str, threshold: float = 0.15) -> tuple:
    """
    Find the most relevant FAQ answer using TF-IDF and cosine similarity.
    Uses a lower threshold for better matching.
    """
    # Preprocess all FAQ questions and user input
    cleaned_faq_questions = [preprocess_text(faq["question"]) for faq in FAQ_DATA]
    cleaned_user_question = preprocess_text(user_question)
    
    # Handle empty input after preprocessing
    if not cleaned_user_question.strip():
        # Try matching based on raw text if preprocessing emptied it
        cleaned_user_question = user_question.lower()
    
    # TF-IDF Vectorization with better parameters
    vectorizer = TfidfVectorizer(
        analyzer='word', 
        ngram_range=(1, 3),  # Added trigrams for better phrase matching
        min_df=1,
        stop_words=None  # We handle stopwords in preprocessing
    )
    
    # Combine FAQ questions with user query
    all_texts = cleaned_faq_questions + [cleaned_user_question]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Extract vectors
    user_vector = tfidf_matrix[-1]
    faq_vectors = tfidf_matrix[:-1]
    
    # Calculate cosine similarities
    similarities = cosine_similarity(user_vector, faq_vectors).flatten()
    
    # Keyword-based boost for better matching
    user_words = set(cleaned_user_question.split())
    for idx, faq in enumerate(FAQ_DATA):
        # Check for keyword matches
        faq_keywords = set(faq.get("keywords", []))
        if user_words.intersection(faq_keywords):
            similarities[idx] += 0.3  # Boost similarity for keyword matches
        
        # Check for direct word overlap
        faq_words = set(cleaned_faq_questions[idx].split())
        word_overlap = len(user_words.intersection(faq_words)) / max(len(faq_words), 1)
        similarities[idx] += word_overlap * 0.2
    
    # Find best match
    max_similarity = float(max(similarities))
    best_match_idx = int(similarities.argmax())
    
    # Lower threshold for refund-related queries
    if 'refund' in user_question.lower() or 'money' in user_question.lower():
        threshold = 0.1  # Even lower threshold for refund questions
    
    # Return response
    if max_similarity >= threshold:
        return FAQ_DATA[best_match_idx]["answer"], max_similarity
    else:
        # Provide helpful suggestions based on keywords
        user_lower = user_question.lower()
        if 'refund' in user_lower or 'money' in user_lower or 'back' in user_lower:
            suggestion = "I see you're asking about refunds. " + FAQ_DATA[6]["answer"]
            return suggestion, max_similarity
        elif 'shipping' in user_lower or 'delivery' in user_lower:
            suggestion = "Regarding shipping: " + FAQ_DATA[1]["answer"]
            return suggestion, max_similarity
        elif 'track' in user_lower or 'where' in user_lower:
            suggestion = "To track your order: " + FAQ_DATA[3]["answer"]
            return suggestion, max_similarity
        else:
            return ("I'm sorry, I couldn't find a precise match for your question. "
                   f"Please try rephrasing or ask about: returns, shipping, tracking, payments, or cancellations."), max_similarity

# =============================================================================
# STREAMLIT UI CONFIGURATION
# =============================================================================
def initialize_session_state():
    """Initialize Streamlit session state variables for chat history."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", 
             "content": "Hello! I'm your FAQ assistant. Ask me about:\n• Returns & Refunds\n• Shipping & Delivery\n• Order Tracking\n• Payment Methods\n• Order Cancellations"}
        ]
    
    if "conversation_count" not in st.session_state:
        st.session_state.conversation_count = 0

def clear_conversation():
    """Reset chat history."""
    st.session_state.messages = [
        {"role": "assistant", 
         "content": "Conversation cleared! How can I help you today? You can ask about refunds, shipping, tracking, payments, or returns."}
    ]
    st.session_state.conversation_count = 0

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="FAQ Chatbot - CodeAlpha Assignment",
        page_icon="🤖",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("ℹ️ About This Chatbot")
        st.markdown("""
        **Domain:** E-commerce & Tech Support
        
        **Capabilities:**
        - Answers FAQs about orders, shipping, payments, and policies
        - Uses NLP (TF-IDF + Cosine Similarity) for intelligent matching
        - Maintains conversation history
        
        **Try asking:**
        - "When will I get my refund?"
        - "What's your return policy?"
        - "How long does shipping take?"
        - "Can I track my order?"
        """)
        
        st.divider()
        
        st.metric("📚 FAQ Database", f"{len(FAQ_DATA)} questions")
        st.metric("💬 Messages", st.session_state.conversation_count)
        
        st.divider()
        
        if st.button("🗑️ Clear Conversation", use_container_width=True, type="primary"):
            clear_conversation()
            st.rerun()
        
        st.divider()
        
        with st.expander("🔧 Technical Details"):
            st.markdown("""
            **NLP Pipeline:**
            1. Smart preprocessing (preserves key terms)
            2. TF-IDF vectorization (unigrams + trigrams)
            3. Cosine similarity with keyword boosting
            4. Threshold-based retrieval (0.15)
            """)
    
    # Main chat interface
    st.title("🤖 FAQ Chatbot Assistant")
    st.caption("Your intelligent Q&A system - Now with better refund handling!")
    
    st.divider()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_query := st.chat_input("Ask me about refunds, shipping, returns, or tracking..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.conversation_count += 1
        
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Finding the best answer..."):
                response, confidence = get_best_response(user_query, threshold=0.15)
            
            st.markdown(response)
            if confidence > 0 and confidence < 0.3:
                st.caption(f"📊 Match confidence: {confidence:.2%}")
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

if __name__ == "__main__":
    main()