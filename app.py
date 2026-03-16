import re
import streamlit as st
import pandas as pd
import plotly.express as px

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Sentiment Analysis",
    page_icon="🛒",
    layout="wide",
)

# ── helper: rule-based predictor (no data needed) ────────────────────────────
POS_WORDS = {
    "good", "great", "excellent", "amazing", "love", "loved", "awesome",
    "perfect", "best", "wonderful", "fantastic", "nice", "happy", "enjoy"
}
NEG_WORDS = {
    "bad", "terrible", "awful", "worst", "hate", "hated", "refund",
    "disappointed", "poor", "waste", "broken", "boring", "problem"
}

def rule_based_predict(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    pos = sum(1 for w in words if w in POS_WORDS)
    neg = sum(1 for w in words if w in NEG_WORDS)
    return "Positive 😊" if pos >= neg else "Negative 😞"

# ── pre-computed results (from our last run) ──────────────────────────────────
RESULTS = pd.DataFrame([
    {"Model": "Majority Class",              "Accuracy": 0.5199, "Precision": 0.5199, "Recall": 1.0000, "F1-score": 0.6841},
    {"Model": "Rule-Based",                  "Accuracy": 0.6492, "Precision": 0.5998, "Recall": 0.9774, "F1-score": 0.7434},
    {"Model": "Logistic Regression (TF-IDF)","Accuracy": 0.8851, "Precision": 0.8859, "Recall": 0.8940, "F1-score": 0.8900},
    {"Model": "SVM - LinearSVC (TF-IDF)",    "Accuracy": 0.8803, "Precision": 0.8824, "Recall": 0.8882, "F1-score": 0.8853},
    {"Model": "PyTorch FFNN (TF-IDF)",       "Accuracy": 0.8800, "Precision": 0.8778, "Recall": 0.8937, "F1-score": 0.8857},
])

# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Model Results", "Try It Yourself"])

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 – Overview
# ═════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("🛒 Amazon Sentiment Analysis")
    st.markdown(
        "Sentiment analysis on **100,000 Amazon product reviews** using "
        "baselines, classical ML, and a neural network."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews",  "100,000")
    col2.metric("Positive Reviews", "51,267")
    col3.metric("Negative Reviews", "48,733")

    st.divider()

    st.subheader("Pipeline")
    st.markdown("""
| Step | Details |
|---|---|
| **Data** | fastText format from Kaggle |
| **Split** | 80,000 train / 20,000 validation |
| **Features** | TF-IDF (10K features, unigrams + bigrams) |
| **Models** | Majority class · Rule-based · Logistic Regression · SVM · PyTorch FFNN |
""")

    st.subheader("Label Distribution")
    label_df = pd.DataFrame({"Label": ["Positive", "Negative"], "Count": [51267, 48733]})
    fig = px.pie(label_df, names="Label", values="Count",
                 color_discrete_sequence=["#4CAF50", "#F44336"])
    st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 – Model Results
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Model Results":
    st.title("📊 Model Results")
    st.markdown("All models evaluated on the **20,000-sample validation set**.")

    # Table
    styled = RESULTS.style.format({
        "Accuracy": "{:.4f}", "Precision": "{:.4f}",
        "Recall": "{:.4f}", "F1-score": "{:.4f}"
    }).highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-score"],
                     color="#d4edda")
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()

    # Bar chart – F1-score
    metric = st.selectbox("Compare by metric", ["F1-score", "Accuracy", "Precision", "Recall"])
    fig = px.bar(
        RESULTS, x="Model", y=metric,
        color="Model", text_auto=".4f",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(showlegend=False, yaxis_range=[0.45, 1.0])
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Best F1:** Logistic Regression (0.8900) — "
        "SVM (0.8853) and PyTorch FFNN (0.8857) are very close behind."
    )

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Try It Yourself
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Try It Yourself":
    st.title("✍️ Try It Yourself")
    st.markdown(
        "Type an Amazon-style review below and see the **Rule-Based** model predict its sentiment in real time."
    )

    user_input = st.text_area("Your review", placeholder="e.g. This product is amazing, I love it!", height=150)

    if st.button("Predict"):
        if user_input.strip():
            result = rule_based_predict(user_input)
            if "Positive" in result:
                st.success(f"Prediction: **{result}**")
            else:
                st.error(f"Prediction: **{result}**")

            # show which keywords matched
            words = re.sub(r"[^a-z\s]", " ", user_input.lower()).split()
            matched_pos = [w for w in words if w in POS_WORDS]
            matched_neg = [w for w in words if w in NEG_WORDS]
            if matched_pos:
                st.markdown(f"Positive keywords found: `{'`, `'.join(matched_pos)}`")
            if matched_neg:
                st.markdown(f"Negative keywords found: `{'`, `'.join(matched_neg)}`")
        else:
            st.warning("Please enter a review first.")
