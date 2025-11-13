import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm

# === PAGE SETUP ===
st.set_page_config(page_title="Portfolio Sentiment Dashboard", layout="wide")
st.title("📊 Portfolio Sentiment Dashboard (FinBERT + VADER)")
st.write("Track sector-level sentiment and compute a Portfolio Sentiment Index (PSI) in real time.")

# === LOAD FINBERT MODEL (cached permanently per server session) ===
@st.cache_resource(show_spinner=True)
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert", torch_dtype="auto"
    )
    return pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)

with st.spinner("🔄 Loading FinBERT model (first run may take ~30s)..."):
    finbert = load_finbert()
st.success("✅ FinBERT model ready and cached!")

# === NEWS FETCH + SENTIMENT (cached, refresh manually) ===
@st.cache_data(ttl=None)
def get_sector_sentiment(api_key):
    analyzer = SentimentIntensityAnalyzer()
    sectors = [
        "Technology sector", "Energy sector", "Financial sector", "Healthcare sector",
        "Consumer discretionary sector", "Industrials sector", "Materials sector",
        "Utilities sector", "Real estate sector", "Communication services sector"
    ]
    results = []

    for sector in tqdm(sectors):
        try:
            url = f"https://newsapi.org/v2/everything?q={sector}&language=en&sortBy=publishedAt&pageSize=30&apiKey={api_key}"
            r = requests.get(url)
            articles = r.json().get("articles", [])
            texts = [(a.get("title") or "") + " " + (a.get("description") or "") for a in articles if a.get("title")]

            # VADER Sentiment
            vader_scores = [analyzer.polarity_scores(t)["compound"] for t in texts]
            avg_vader = round(sum(vader_scores)/len(vader_scores), 3) if vader_scores else None

            # FinBERT Sentiment
            finbert_scores = []
            for text in texts:
                try:
                    result = finbert(text[:512])[0]
                    label = result["label"]
                    score = result["score"] if label == "positive" else -result["score"] if label == "negative" else 0
                    finbert_scores.append(score)
                except Exception:
                    continue

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

    df = pd.DataFrame(results)
    finbert_weight, vader_weight = 0.7, 0.3
    df["Weighted Sentiment"] = finbert_weight * df["Sentiment (FinBERT)"] + vader_weight * df["Sentiment (VADER)"]
    df = df.sort_values(by="Weighted Sentiment", ascending=False).reset_index(drop=True)

    mean_sent, std_sent = df["Weighted Sentiment"].mean(), df["Weighted Sentiment"].std()
    df["Z-Score Sentiment"] = (df["Weighted Sentiment"] - mean_sent) / std_sent if std_sent != 0 else 0
    return df

# === SIDEBAR CONTROLS ===
st.sidebar.header("⚙️ Controls")

# Load API key securely
API_KEY = st.secrets.get("NEWS_API_KEY")
if not API_KEY:
    st.error("⚠️ Missing News API Key! Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

# Refresh button (only clears news cache)
if st.sidebar.button("🔄 Refresh News & Sentiment"):
    st.cache_data.clear()
    st.experimental_rerun()

# === GET (CACHED) SECTOR DATA ===
df_sentiment = get_sector_sentiment(API_KEY)

# === PORTFOLIO WEIGHTS ===
st.sidebar.header("📈 Set Portfolio Weights (%)")
weights = {sector: st.sidebar.number_input(f"{sector}", min_value=0.0, max_value=100.0, value=0.0)
           for sector in df_sentiment["Sector"]}

df_weights = pd.DataFrame(list(weights.items()), columns=["Sector", "Weight (%)"])
total = df_weights["Weight (%)"].sum()
if total != 0:
    df_weights["Weight (%)"] = df_weights["Weight (%)"] / total * 100

# === SENTIMENT CALCULATIONS ===
market_sentiment = df_sentiment["Weighted Sentiment"].mean()
df_sentiment["Adj Sentiment"] = df_sentiment["Weighted Sentiment"] - market_sentiment

merged = pd.merge(df_sentiment, df_weights, on="Sector", how="inner")
merged["Weighted Sentiment (Abs)"] = merged["Weighted Sentiment"] * (merged["Weight (%)"]/100)
merged["Weighted Sentiment (Adj)"] = merged["Adj Sentiment"] * (merged["Weight (%)"]/100)
merged["Weighted Sentiment (Z)"]   = merged["Z-Score Sentiment"] * (merged["Weight (%)"]/100)

psi_absolute = merged["Weighted Sentiment (Abs)"].sum()
psi_relative = merged["Weighted Sentiment (Adj)"].sum()
psi_zscore = merged["Weighted Sentiment (Z)"].sum()

# === SENTIMENT LABEL FUNCTION ===
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

# === DISPLAY RESULTS ===
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

# === VISUALIZATION ===
st.subheader("📊 Sector Sentiment (FinBERT-Weighted Z-Score)")
fig, ax = plt.subplots()
ax.barh(merged["Sector"], merged["Z-Score Sentiment"], color='skyblue')
ax.axvline(0, color='black', linestyle='--')
ax.set_xlabel("Z-Score (Relative Sentiment)")
ax.set_title("Sector Sentiment (FinBERT-Weighted)")
st.pyplot(fig)

# === SENTIMENT BACKTEST (TECH & MARKET vs S&P 500) ===
# === LONG-TERM SENTIMENT vs LOG(S&P 500) ===
import datetime
import numpy as np
import yfinance as yf

today = datetime.date.today().isoformat()
HIST_FILE = "sentiment_history.csv"

# --- Extract today's market sentiment ---
market_sent = df_sentiment["Weighted Sentiment"].mean()

df_new = pd.DataFrame({
    "Date": [today],
    "Market Sentiment": [market_sent]
})

# --- Append / update sentiment history ---
if os.path.exists(HIST_FILE):
    df_hist = pd.read_csv(HIST_FILE)
    df_hist = pd.concat([df_hist, df_new]).drop_duplicates(subset=["Date"], keep="last")
else:
    df_hist = df_new

df_hist["Date"] = pd.to_datetime(df_hist["Date"])
df_hist.to_csv(HIST_FILE, index=False)

# --- Load long-term S&P 500 data ---
sp500 = yf.download("^GSPC", start=df_hist["Date"].min(), end=today)

if sp500.empty:
    st.warning("⚠️ No S&P 500 data returned.")
else:
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)
    sp500 = sp500.reset_index()
    if "Adj Close" in sp500.columns:
        sp500.rename(columns={"Adj Close": "S&P500"}, inplace=True)
    elif "Close" in sp500.columns:
        sp500.rename(columns={"Close": "S&P500"}, inplace=True)
    sp500["Date"] = pd.to_datetime(sp500["Date"])

    # --- Merge and compute log prices ---
    df_merged = pd.merge(df_hist, sp500[["Date", "S&P500"]], on="Date", how="inner")
    df_merged["Log S&P500"] = np.log(df_merged["S&P500"])

    # --- Scale sentiment for plotting (match range roughly) ---
    sentiment_scaled = (df_merged["Market Sentiment"] - df_merged["Market Sentiment"].mean()) * 2000 + df_merged["Log S&P500"].mean()

    # --- Plot ---
    st.subheader("📈 Long-Term Market Sentiment vs log(S&P 500)")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_merged["Date"], df_merged["Log S&P500"], label="log(S&P 500)", color="green")
    ax.plot(df_merged["Date"], sentiment_scaled, label="Market Sentiment (scaled)", color="blue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Price / Scaled Sentiment")
    ax.legend()
    ax.set_title("Market Sentiment vs log(S&P 500) Over Time")
    st.pyplot(fig)

    # --- Correlation (direct) ---
    corr = df_merged["Market Sentiment"].corr(df_merged["Log S&P500"])
    st.markdown(f"**📊 Correlation between Market Sentiment and log(S&P 500):** `{corr:.3f}`")





