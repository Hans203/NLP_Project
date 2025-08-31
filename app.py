import streamlit as st
import pandas as pd
import re
import nltk
from nltk import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
from textblob import TextBlob
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()


# ---- Aspect Definitions ----
aspects = {
    "Engine/Performance": ["engine", "powerful", "performance", "speed"],
    "Mileage/Fuel": ["mileage", "fuel", "consumption", "efficiency"],
    "Comfort/Interior": ["comfort", "seat", "interior", "space"],
    "Design/Style": ["design", "look", "style", "exterior"],
    "Price/Value": ["price", "cost", "expensive", "worth"],
    "Service/Maintenance": ["service", "maintenance", "support", "parts"]
}

# ---- Functions ----
def get_aspect(sentence, model, aspects):
    tokens = word_tokenize(sentence.lower())
    words = [t for t in tokens if t in model.wv.key_to_index]
    max_sim, best_aspect = 0, None
    for aspect, keywords in aspects.items():
        for keyword in keywords:
            for word in words:
                if word in model.wv and keyword in model.wv:
                    sim = model.wv.similarity(word, keyword)
                    if sim > max_sim:
                        max_sim, best_aspect = sim, aspect
    return best_aspect

def get_sentiment(sentence):
    polarity = TextBlob(sentence).sentiment.polarity
    return "positive" if polarity > 0 else "negative"

def get_results(brand, review, model):
    sentences = sent_tokenize(review)
    results = []
    neg_sens = []
    for sentence in sentences:
        aspect = get_aspect(sentence, model, aspects)
        sentiment = get_sentiment(sentence)
        if sentiment == "negative":
            neg_sens.append((brand, aspect, sentence))
        results.append((brand, aspect, sentiment))
    return results, neg_sens

# ---- Gemini Setup ----
def setup_gemini():
    # 🔑 Hardcoded API key
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")

def summarize_brand_problems(model, brand, reviews):
    all_reviews = " ".join(reviews)
    prompt = f"""
    Analyze these negative customer reviews for {brand} and summarize the main problems customers are facing.

    Reviews: {all_reviews}

    Please provide a clear, concise summary of the top 3-5 problems customers mention most often.
    Keep it simple and actionable.
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating summary: {e}"

# ---- Load Pretrained CBOW Model ----
@st.cache_resource
def load_cbow_model():
    return Word2Vec.load("car_reviews_cbow.model")

w2v_model = load_cbow_model()

# ---- Streamlit App ----
st.title("Aspect Based Sentiment Analysis for Car Reviews")

mode = st.radio("Choose Analysis Mode:", ["Demo: Car Reviews for 5 Brands", "Upload Reviews for Analysis"  ])

if mode == "Demo: Car Reviews for 5 Brands":

    summary = pd.read_csv("summary.csv")  # your saved summary CSV
    st.write("### Sentiment Summary by Brand & Aspect")
    st.dataframe(summary)
    for brand in summary['brand'].unique():
        with st.expander(f"🔍 Insights for {brand}"):
            brand_data = summary[summary['brand'] == brand]
            for _, row in brand_data.iterrows():
                pos_pct = row['positive_pct']
                neg_pct = row['negative_pct']
                aspect = row['aspect']
                st.write(f"🟢 {pos_pct:.1f}% positive " f"and 🔴 {neg_pct:.1f}% negative reviews for **{aspect}**.")

    # Precomputed LLM summaries
    precomputed_df = pd.read_csv("brand_problems_summary.csv")  # your saved CSV
    st.write("### General Problems faced By Customers")
    for _, row in precomputed_df.iterrows():
        st.subheader(f"{row['Brand']}")
        st.write(row['Problem_Summary'])


elif mode == "Upload Reviews for Analysis":
    uploaded_file = st.file_uploader("Upload a CSV file with car reviews", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("### Preview of Uploaded Data")
        st.dataframe(df.head())
        # Convert all column names to lowercase
        df.columns = df.columns.str.lower()
        brand_input = st.text_input("Enter the brand name for these reviews:")

        if brand_input:
            # Add brand_name column
            df["brand"] = brand_input  

            # Preprocess text
            df['review'] = df['review'].astype(str).apply(lambda x: re.sub(r'[^A-Za-z\s]', '', x).lower())
            with st.spinner("Analysing reviews..."):
            # Run analysis
                results, neg_sens = [], []
                for idx, row in df.iterrows():
                    brand = row['brand']
                    review = row['review']
                    r, n = get_results(brand, review, w2v_model)
                    results.extend(r)
                    if n:
                        neg_sens.extend(n)

            df_result = pd.DataFrame(results, columns=["brand", "aspect", "sentiment"])
            summary = df_result.groupby(["brand", "aspect", "sentiment"]).size().unstack(fill_value=0)
            summary["total"] = summary.sum(axis=1)
            summary["positive_pct"] = (summary.get("positive", 0) / summary["total"]) * 100
            summary["negative_pct"] = (summary.get("negative", 0) / summary["total"]) * 100

            st.write("### Sentiment Summary by Brand & Aspect")
            st.dataframe(summary)
            summary = summary.reset_index()
            # Natural language insights
            for brand in summary['brand'].unique():
                with st.expander(f"🔍 Insights for {brand}"):
                    brand_data = summary[summary['brand'] == brand]
                    for _, row in brand_data.iterrows():
                        pos_pct = row['positive_pct']
                        neg_pct = row['negative_pct']
                        aspect = row['aspect']
                        st.write(f"🟢 {pos_pct:.1f}% positive " f"and 🔴 {neg_pct:.1f}% negative reviews for **{aspect}**.")

            # Negative reviews
            if neg_sens:
                negative_reviewsdf = pd.DataFrame(neg_sens, columns=["brand", "aspect", "negative_review"])
                st.write("### Extracted Negative Reviews")
                st.dataframe(negative_reviewsdf.head())

                # LLM Summarization
                st.write("### General Problems faced By Customers")
                model = setup_gemini()
                for brand in negative_reviewsdf['brand'].unique():
                    brand_reviews = negative_reviewsdf[negative_reviewsdf['brand'] == brand]['negative_review'].tolist()
                    summary_text = summarize_brand_problems(model, brand, brand_reviews)
                    st.subheader(f"{brand}")
                    st.write(summary_text)

                st.success("LLM problem summaries generated successfully!")
