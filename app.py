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

# Importing Libraries
import streamlit as st
import tensorflow as tf
import keras
import pandas as pd
import numpy as np

# Loading Model
model = keras.models.load_model('model.keras')

# Creating the Page

# Container for Heading


# Container for About the Project
with st.container():

    # an expander for about project section
    with st.expander('**:blue[WHAT IS THIS ?]**', expanded=True):

        st.write('**:green[Heart🫀 Disease Predictor] is a supervised DL model 👍🏻dignosing👎🏻 :green[Heart🫀 diseases].**')
        st.write('**An Artificial Neural Network (ANN) is working behind the model.**')

# Container for Patients Details
with st.container():

    # Output container
    with st.container(border=False):
        output = st.empty()
 
    # age, sex, cp columns
    age_col, sex_col, cp_col = st.columns(3)
        
    with age_col:
        age = st.text_input('**:blue[AGE]**', placeholder='Enter Age here..')
        if age:
            age = np.int64(age)
        
    with sex_col:
        sex = st.selectbox('**:blue[GENDER]**', options=['Male', 'Female'])
        if sex == 'Male':
            sex = 1
        else: 
            sex = 0
        
    with cp_col:
        cp = st.selectbox('**:blue[CP]**', options=[0,1,2,3,4])
        
    # trestbps, chol, fbs columns
    trestbps_col, chol_col, fbs_col = st.columns(3)

    with trestbps_col:
        trestbps = st.text_input('**:blue[TRESTBPS]**', placeholder='Enter between 94 to 200')
        if trestbps:
            trestbps = np.int64(trestbps)
        
    with chol_col:
        chol = st.text_input('**:blue[CHOL]**', placeholder='Enter between 126 to 564')
        if chol:
            chol = np.int64(chol) 
        
    with fbs_col:
        fbs = st.selectbox('**:blue[FBS]**', options=[True, False])
        if fbs == True:
            fbs = 1
        else:
            fbs = 0

    # restecg, thalach, exang columns
    restecg_col, thalach_col, exang_col = st.columns(3)

    with restecg_col:
        restecg = st.selectbox('**:blue[RESTECG]**', options=[0, 1, 2])

    with thalach_col:
        thalach = st.text_input('**:blue[THALACH]**', placeholder='Enter between 71 to 202')
        if thalach:
            thalach = np.int64(thalach)

    with exang_col:
        exang = st.selectbox('**:blue[EXANG]**', options=['Yes', 'No'])
        if exang == 'Yes':
            exang = 1
        else:
            exang = 0

    # oldpeak, slope, ca columns
    oldpeak_col, slope_col, ca_col = st.columns(3)

    with oldpeak_col:
        oldpeak = st.text_input('**:blue[OLDPEAK]**', placeholder='Enter between 0.1 to 6.2')
        if oldpeak:
            oldpeak = np.float32(oldpeak)

    with slope_col:
        slope = st.selectbox('**:blue[SLOPE]**', options=[1,2,3])

    with ca_col:
        ca = st.selectbox('**:blue[CA]**', options=[0,1,2,3])
        
    # thal column
    thal_col, submit_col = st.columns([1,2])

    with thal_col:
        thal = st.selectbox('**:blue[THAL]**', options=['1','2','fixed', 'normal', 'reversible'])
        
    sample = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang, 
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
            }
    if age and trestbps and chol and thalach and oldpeak:
        input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
        predictions = model.predict(input_dict)
        output.info(f'**PERCENTAGE OF HEART🫀 DISEASE IS :blue[{100 * predictions[0][0]:.2f} %]**')

