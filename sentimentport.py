import requests
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from tqdm import tqdm
import streamlit as st
import matplotlib.pyplot as plt

# === SETUP ===
# Get your API key from Streamlit Secrets (for security)
import os
API_KEY = os.getenv("NEWS_API_KEY")  # Add in Streamlit Cloud under "Settings → Secrets"

analyzer = SentimentIntensityAnalyzer()
from transformers import AutoTokenizer, AutoModelForSequenceClassification

with st.spinner("Loading FinBERT model..."):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    finbert = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)
st.success("✅ FinBERT model loaded successfully!")



# Define your sectors
sectors = [
    "Technology sector",
    "Energy sector",
    "Financial sector",
    "Healthcare sector",
    "Consumer discretionary sector",
    "Industrials sector",
    "Materials sector",
    "Utilities sector",
    "Real estate sector",
    "Communication services sector"
]

results = []

# === LOOP THROUGH SECTORS ===
for sector in tqdm(sectors):
    try:
        # Fetch latest headlines for this sector
        url = f"https://newsapi.org/v2/everything?q={sector}&language=en&sortBy=publishedAt&pageSize=30&apiKey={API_KEY}"
        r = requests.get(url)
        articles = r.json().get("articles", [])

        # Extract text (title + description)
        texts = [
            (a.get("title") or "") + " " + (a.get("description") or "")
            for a in articles if a.get("title")
        ]

        # === VADER Sentiment ===
        vader_scores = [analyzer.polarity_scores(t)["compound"] for t in texts]
        avg_vader = round(sum(vader_scores)/len(vader_scores), 3) if vader_scores else None

        # === FinBERT Sentiment ===
        finbert_scores = []
        for text in texts:
            try:
                result = finbert(text[:512])[0]  # limit to 512 tokens
                label = result["label"]
                score = result["score"] if label == "positive" else -result["score"] if label == "negative" else 0
                finbert_scores.append(score)
            except Exception:
                continue  # skip problematic headlines

        avg_finbert = round(np.mean(finbert_scores), 3) if finbert_scores else None

        results.append({
            "Sector": sector,
            "Sentiment (VADER)": avg_vader,
            "Sentiment (FinBERT)": avg_finbert,
            "Headlines": len(texts)
        })

    except Exception as e:
        results.append({
            "Sector": sector,
            "Sentiment (VADER)": None,
            "Sentiment (FinBERT)": None,
            "Headlines": 0
        })
        print(f"Error processing {sector}: {e}")

# === BUILD DATAFRAME ===
df_sentiment = pd.DataFrame(results)

# === Apply weighted FinBERT + VADER sentiment ===
finbert_weight = 0.7
vader_weight = 0.3

df_sentiment["Weighted Sentiment"] = (
    finbert_weight * df_sentiment["Sentiment (FinBERT)"] +
    vader_weight * df_sentiment["Sentiment (VADER)"]
)

# === Sort and normalize ===
df_sentiment = df_sentiment.sort_values(by="Weighted Sentiment", ascending=False).reset_index(drop=True)
mean_sent = df_sentiment["Weighted Sentiment"].mean()
std_sent = df_sentiment["Weighted Sentiment"].std()
df_sentiment["Z-Score Sentiment"] = (
    (df_sentiment["Weighted Sentiment"] - mean_sent) / std_sent if std_sent != 0 else 0
)

# === STREAMLIT DASHBOARD ===
st.title("📊 Portfolio Sentiment Dashboard (FinBERT + VADER)")
st.write("Track sector-level sentiment and compute a Portfolio Sentiment Index (PSI) in real time.")

# === Sidebar weights ===
st.sidebar.header("Set Portfolio Weights (%)")
weights = {}
for sector in df_sentiment["Sector"]:
    weights[sector] = st.sidebar.number_input(f"{sector}", min_value=0.0, max_value=100.0, value=0.0)

df_weights = pd.DataFrame(list(weights.items()), columns=["Sector", "Weight (%)"])
total = df_weights["Weight (%)"].sum()
if total != 0:
    df_weights["Weight (%)"] = df_weights["Weight (%)"] / total * 100

# === Sentiment calculations ===
market_sentiment = df_sentiment["Weighted Sentiment"].mean()
df_sentiment["Adj Sentiment"] = df_sentiment["Weighted Sentiment"] - market_sentiment

merged = pd.merge(df_sentiment, df_weights, on="Sector", how="inner")
merged["Weighted Sentiment (Abs)"] = merged["Weighted Sentiment"] * (merged["Weight (%)"]/100)
merged["Weighted Sentiment (Adj)"] = merged["Adj Sentiment"] * (merged["Weight (%)"]/100)
merged["Weighted Sentiment (Z)"]   = merged["Z-Score Sentiment"] * (merged["Weight (%)"]/100)

psi_absolute = merged["Weighted Sentiment (Abs)"].sum()
psi_relative = merged["Weighted Sentiment (Adj)"].sum()
psi_zscore = merged["Weighted Sentiment (Z)"].sum()

# === Label interpretation ===
def sentiment_label(psi):
    if psi >= 0.3:
        return "🚀 Strongly Bullish"
    elif psi >= 0.1:
        return "📈 Mildly Bullish"
    elif psi > -0.1:
        return "😐 Neutral"
    elif psi > -0.3:
        return "📉 Mildly Bearish"
    else:
        return "💀 Strongly Bearish"

# === Display results ===
st.subheader("📄 Portfolio Sentiment Summary")
st.dataframe(merged[[
    "Sector", "Weighted Sentiment", "Adj Sentiment", "Weight (%)",
    "Weighted Sentiment (Abs)", "Weighted Sentiment (Adj)", "Z-Score Sentiment"
]])

st.markdown(f"""
**🌍 Market Sentiment (Overall Tone):** {market_sentiment:.3f}  
**💼 Portfolio Sentiment Index (Absolute):** {psi_absolute:.3f} → {sentiment_label(psi_absolute)}  
**⚖️ Portfolio Sentiment Index (Relative):** {psi_relative:.3f} → {sentiment_label(psi_relative)}  
**📏 Portfolio Sentiment Index (Z-Score):** {psi_zscore:.3f} → {sentiment_label(psi_zscore)}  
""")

# === Visualization ===
st.subheader("📊 Sector Sentiment (FinBERT-Weighted Z-Score)")
fig, ax = plt.subplots()
ax.barh(merged["Sector"], merged["Z-Score Sentiment"], color='skyblue')
ax.axvline(0, color='black', linestyle='--')
ax.set_xlabel("Z-Score (Relative Sentiment)")
ax.set_title("Sector Sentiment (FinBERT-Weighted)")
st.pyplot(fig)
