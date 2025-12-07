import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import json
from datetime import datetime
from dotenv import load_dotenv

# --- CONFIGURATION ---

# 1. Load Environment Variables from .env file
load_dotenv()

# 2. Page Config
st.set_page_config(page_title="Fynd AI Feedback System", layout="wide")

# 3. Securely Retrieve API Key
api_key = os.getenv("GEMINI_API_KEY")

# Fallback: Try Streamlit Secrets (for Cloud Deployment)
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except FileNotFoundError:
        st.error("❌ CRITICAL ERROR: API Key not found.")
        st.info("Please create a '.env' file with 'GEMINI_API_KEY=your_key' in the project folder.")
        st.stop()

# 4. Configure Gemini
try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Error configuring API: {e}")

# File to store data
DATA_FILE = "reviews_data.csv"

# --- BACKEND FUNCTIONS ---

def load_data():
    """Loads reviews from CSV or creates an empty DataFrame."""
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        columns = ["timestamp", "user_rating", "review_text", "ai_response_user", "ai_summary", "ai_actions"]
        return pd.DataFrame(columns=columns)

def save_data(new_entry):
    """Appends a new review to the CSV."""
    df = load_data()
    new_df = pd.DataFrame([new_entry])
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

def process_review_with_ai(review_text, rating):
    """
    Uses Gemini to analyze the review. 
    Enforces JSON output for reliability.
    """
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    You are a customer service AI manager.
    Analyze the following customer review:
    
    Review: "{review_text}"
    Rating: {rating}/5
    
    Return a valid JSON object with exactly these 3 keys:
    1. "user_response": A polite, empathetic response to the customer (max 2 sentences).
    2. "summary": A very short 5-word summary of the core issue or praise.
    3. "actions": A short string listing recommended internal actions (e.g., "Check inventory, Train staff").
    """
    
    try:
        # Force JSON response structure
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        return json.loads(response.text)

    except Exception as e:
        # Print error to Terminal for debugging
        print(f"\n❌ AI ERROR: {e}\n") 
        
        return {
            "user_response": "Thank you for your feedback!",
            "summary": "AI Processing Error",
            "actions": "Please review logs"
        }

# --- UI: SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["User Dashboard", "Admin Dashboard"])

# --- UI: USER DASHBOARD ---
if page == "User Dashboard":
    st.title("📝 Submit Your Feedback")
    st.write("We value your opinion. Please rate your experience.")
    
    with st.form("review_form"):
        rating = st.slider("Rating", 1, 5, 5)
        review_text = st.text_area("Your Review", placeholder="Tell us what you liked or disliked...")
        submitted = st.form_submit_button("Submit Review")
        
        if submitted and review_text:
            with st.spinner("Analyzing your feedback..."):
                # 1. Call LLM
                ai_results = process_review_with_ai(review_text, rating)
                
                # 2. Prepare Data
                new_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "user_rating": rating,
                    "review_text": review_text,
                    "ai_response_user": ai_results.get("user_response", "Thank you!"),
                    "ai_summary": ai_results.get("summary", "N/A"),
                    "ai_actions": ai_results.get("actions", "N/A")
                }
                
                # 3. Save Data
                save_data(new_data)
                
                # 4. Show Output
                st.success("Feedback Submitted!")
                st.info(f"**Automated Response:** {new_data['ai_response_user']}")
                
        elif submitted and not review_text:
            st.warning("Please write a review before submitting.")

# --- UI: ADMIN DASHBOARD ---
elif page == "Admin Dashboard":
    st.title("📊 Feedback Analytics & Insights")
    
    df = load_data()
    
    if df.empty:
        st.info("No reviews submitted yet.")
    else:
        # Top Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", len(df))
        col1.metric("Avg Rating", f"{df['user_rating'].mean():.2f}")
        
        # Simple Sentiment Logic
        positive_count = len(df[df['user_rating'] >= 4])
        col2.metric("Positive Feedback", f"{positive_count}")
        col3.metric("Critical Feedback", f"{len(df) - positive_count}")

        st.divider()

        # Detailed View
        st.subheader("Live Feed")
        
        # Display latest first
        for index, row in df.iloc[::-1].iterrows():
            with st.expander(f"{row['timestamp']} | Rating: {row['user_rating']}⭐ | {row['ai_summary']}"):
                col_a, col_b = st.columns([1, 1])
                
                with col_a:
                    st.markdown("**User Review:**")
                    st.write(f"_{row['review_text']}_")
                
                with col_b:
                    st.markdown("**Recommended Actions:**")
                    st.info(row['ai_actions'])
                    st.caption(f"Auto-reply sent: {row['ai_response_user']}")

        # Raw Data Option
        st.divider()
        with st.expander("View Raw Data CSV"):
            st.dataframe(df)