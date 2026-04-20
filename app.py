import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()
from model import load_data, train_model

st.set_page_config(page_title="Customer Segmentation Analyzer", layout="wide")

# ---------------------------
# TITLE
# ---------------------------

st.title("Customer Intelligence Dashboard")
st.caption("AI-driven insights into consumer behavior")


# ---------------------------
# LOAD DATASET
# ---------------------------

df = load_data()

# ---------------------------
# MODEL SETTINGS
# ---------------------------

st.markdown("### ⚙️ Model Controls")

col1, col2 = st.columns([2,1])

with col1:
    k = st.slider("Number of clusters", 2, 10, 5)

with col2:
    st.markdown("""
    <div style="
        background:#181818;
        padding:15px;
        border-radius:10px;
        text-align:center;
        border:1px solid #1DB954;">
        <b>Active Clusters</b><br>
        <span style="font-size:22px;color:#1DB954;">{}</span>
    </div>
    """.format(k), unsafe_allow_html=True)

df, model, scaler = train_model(df, k)

# ---------------------------
# DATASET OVERVIEW
# ---------------------------

st.header("📊 Business Snapshot")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Average Income", round(df["Annual Income (k$)"].mean(),1))
col3.metric("Average Spending Score", round(df["Spending Score (1-100)"].mean(),1))

st.dataframe(df.head())

# ---------------------------
# CUSTOMER PERSONAS
# ---------------------------

persona_map = {
    0: "Affluent Spenders",
    1: "High-Value Targets",
    2: "Impulse Driven",
    3: "Cost-Conscious",
    4: "Balanced Consumers",
    5: "Occasional Buyers",
    6: "Elite Clients",
    7: "Deal Seekers",
    8: "Trend Followers",
    9: "Bulk Purchasers"
}

df["Persona"] = df["Cluster"].map(lambda x: persona_map.get(x, "Customer"))

st.header("🎭 Audience Segments")

persona_counts = df["Persona"].value_counts()

st.bar_chart(persona_counts)

# ---------------------------
# SEGMENTATION MAP
# ---------------------------
st.subheader("Audience Segments Map")

fig, ax = plt.subplots(figsize=(7,5))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Persona",
    palette="Set2",
    s=100,
    edgecolor="white",
    linewidth=1,
    ax=ax
)

ax.set_title("Audience Distribution Map", fontsize=14, fontweight="bold")

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Persona",
    palette="tab10",
    s=80,
    edgecolor="black",
    linewidth=0.5,
    ax=ax
)

ax.set_title("Audience Segments", fontsize=12)
ax.set_xlabel("Income")
ax.set_ylabel("Spending Score")

plt.tight_layout()

st.pyplot(fig, use_container_width=False)

# ---------------------------
# SEGMENT EXPLORER
# ---------------------------

st.header("Segment Deep Dive")

persona_choice = st.selectbox(
    "Select a customer persona",
    df["Persona"].unique()
)

cluster_data = df[df["Persona"] == persona_choice]

st.write(f"Customers in this group: **{len(cluster_data)}**")

st.dataframe(cluster_data.head())

st.subheader("Segment Statistics")

stats = cluster_data[["Age","Annual Income (k$)", "Spending Score (1-100)"]].mean()

st.write(stats)

# ---------------------------
# CUSTOMER BEHAVIOR SIMULATOR
# ---------------------------

st.header("🔮 Predict Customer Profile")

col1, col2 = st.columns(2)

income = col1.slider("Annual Income (k$)", 10, 150, 60)
spending = col2.slider("Spending Score", 1, 100, 50)

sample = scaler.transform([[income, spending]])

prediction = model.predict(sample)[0]

persona = persona_map.get(prediction, "Customer")

st.markdown(f"""
<div style="
    padding:15px;
    border-radius:10px;
    background:#f0f4ff;
    font-weight:600;
    text-align:center;">
Predicted Segment: {persona}
</div>
""", unsafe_allow_html=True)

# ---------------------------
# MARKETING STRATEGY
# ---------------------------

st.header("Strategy Insights")

strategies = {
    "💎 Luxury Shopper": "Offer premium memberships and exclusive products.",
    "🎯 Target Customer": "Provide personalized discounts to increase spending.",
    "🛍 Impulsive Buyer": "Use flash sales and limited-time offers.",
    "💰 Budget Shopper": "Promote affordable bundles and deals.",
    "🧍 Average Customer": "Recommend popular and trending products.",
    "🧑 Occasional Shopper": "Send reminder campaigns and loyalty points.",
    "👑 Premium Elite": "Invite to VIP events and luxury launches.",
    "🛒 Discount Hunter": "Highlight deals, coupons, and seasonal sales.",
    "🎉 Trend Shopper": "Promote new arrivals and trending items.",
    "📦 Bulk Buyer": "Offer bundle discounts and wholesale deals."
}

selected_persona = st.selectbox(
    "Choose persona for marketing strategy",
    list(strategies.keys())
)

st.info(strategies[selected_persona])