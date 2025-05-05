import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
import streamlit as st

# ─── Config & CSS ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="❤️ Heart Disease Dashboard",
    page_icon="❤️",
    layout="wide"
)

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Custom CSS
st.markdown("""
    <style>
        /* Global font */
        body, .css-1d391kg {
            font-family: 'Roboto', sans-serif;
        }
        /* Title styling */
        .big-title {
            font-size: 2.5rem !important;
            color: #e63946 !important;
            font-weight: 700 !important;
        }
        /* Sidebar header */
        .sidebar .css-hi6a2p {
            font-size: 1.25rem !important;
            color: #457b9d !important;
        }
        /* Download button */
        .stDownloadButton>button {
            background-color: #e63946 !important;
            color: #fff !important;
            border-radius: 8px !important;
        }
    </style>
""", unsafe_allow_html=True)

# ─── Data Loading & Preprocess ─────────────────────────────────────────────────
# Create visuals folder
os.makedirs("visuals", exist_ok=True)

df = pd.read_csv('data/heart.csv')

# Fill missing values
for col in ['trestbps', 'chol', 'thalch', 'oldpeak']:
    df[col].fillna(df[col].mean(), inplace=True)
for col in ['fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Map target and standardize cp
df['target'] = df['num'].apply(lambda x: 'Heart Disease' if x > 0 else 'No Heart Disease')
df['cp'] = df['cp'].str.lower().map({
    'typical angina': 'Typical Angina',
    'atypical angina': 'Atypical Angina',
    'non-anginal': 'Non-anginal Pain',
    'asymptomatic': 'Asymptomatic'
})

# ─── Dashboard Header ──────────────────────────────────────────────────────────
st.markdown('<div class="big-title">❤️ Heart Disease Analysis Dashboard</div>', unsafe_allow_html=True)
st.write("Explore interactive charts and filter the data in real time.")

# ─── Interactive Plotly Charts ──────────────────────────────────────────────────

# 1) Pie Chart: Target distribution
st.plotly_chart(
    go.Figure(
        go.Pie(
            labels=df['target'].value_counts().index,
            values=df['target'].value_counts().values,
            hole=0.4
        )
    ).update_layout(
        title="Heart Disease Distribution", 
        template="plotly_dark",
        width=800, height=500
    ), 
    use_container_width=True
)

# 2) Histogram: Age by target
st.plotly_chart(
    px.histogram(
        df, x="age", color="target", 
        nbins=25, barmode="stack", 
        title="Age Distribution by Heart Disease",
        template="plotly_dark",
        height=500
    ).update_traces(opacity=0.8),
    use_container_width=True
)

# 3) Scatter Matrix: Pairplot
st.plotly_chart(
    px.scatter_matrix(
        df, dimensions=["age", "trestbps", "chol", "thalch"], 
        color="target", title="Feature Relationships",
        template="plotly_dark", height=700
    ).update_layout(margin=dict(l=10, r=10, t=60, b=10)),
    use_container_width=True
)

# 4) Resting BP histogram
st.plotly_chart(
    px.histogram(
        df, x="trestbps",
        title="Resting Blood Pressure Distribution",
        template="plotly_dark", nbins=30, height=400
    ).update_traces(marker_line_width=1, marker_line_color="white"),
    use_container_width=True
)

# 5) Age histogram with annotation
fig5 = px.histogram(
    df, x="age", color="target", nbins=25,
    title="Age Distribution with Annotation",
    template="plotly_dark", height=500
)
fig5.add_annotation(
    x=65, y=0.035, 
    text="Peak Age for Heart Disease", 
    showarrow=True, arrowhead=2, arrowsize=1, ax=0, ay=-50
)
st.plotly_chart(fig5, use_container_width=True)

# ─── Sidebar Filters & Data ─────────────────────────────────────────────────────
st.sidebar.header("🔍 Filter Data")
age_range = st.sidebar.slider(
    "Select Age Range",
    int(df['age'].min()), int(df['age'].max()),
    (40, 70)
)
filtered = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]

st.sidebar.download_button(
    "📥 Download Filtered CSV",
    data=filtered.to_csv(index=False),
    file_name="filtered_heart_data.csv",
    mime="text/csv"
)

# Show filtered data in an expander
with st.expander(f"Filtered Data ({len(filtered)} rows)"):
    st.dataframe(filtered.style.highlight_max(axis=0))

# Show counts for filtered data
st.markdown(f"**Heart Disease counts in filtered data:**")
st.bar_chart(filtered['target'].value_counts())

