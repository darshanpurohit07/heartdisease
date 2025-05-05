# Heart Disease Analysis Dashboard

## Objective
The goal of this mini-project is to develop an interactive, visually rich dashboard for analyzing a heart disease dataset using Python and Streamlit. The main objectives are:

- Identify statistical patterns in medical attributes related to heart disease.
- Explore correlations between age, cholesterol levels, chest pain type, and maximum heart rate.
- Leverage interactive visualization tools to present insights to non-technical users.
- Use Python libraries such as `pandas`, `seaborn`, `matplotlib`, and `plotly` for data analysis and visualization.
- Build a web-based UI using Streamlit for real-time exploration of the dataset.

## Problem Statement
Heart disease is one of the leading causes of death worldwide, affecting millions every year. Early identification of heart disease risk factors through clinical patterns in patient data can assist in better treatment planning and preventive healthcare strategies.

However, raw data can often be overwhelming and hard to interpret without a data science background. The goal of this project is to transform the dataset into an interactive and insightful tool for both healthcare professionals and the general public.

## Dataset Description
The dataset used is a widely recognized heart disease dataset, often utilized in medical and machine learning research. It consists of 303 patient records, each described by 14 attributes.

### Features:
| Feature      | Type               | Description                                                                 |
|--------------|--------------------|-----------------------------------------------------------------------------|
| `age`        | Numerical          | Patient's age in years                                                       |
| `sex`        | Binary             | 1 = Male, 0 = Female                                                         |
| `cp`         | Categorical (0-3)   | Chest pain type (0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic) |
| `trestbps`   | Numerical          | Resting blood pressure (in mm Hg)                                            |
| `chol`       | Numerical          | Serum cholesterol in mg/dl                                                   |
| `fbs`        | Binary             | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)                       |
| `restecg`    | Categorical (0-2)   | Resting ECG results (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy) |
| `thalach`    | Numerical          | Maximum heart rate achieved                                                  |
| `exang`      | Binary             | Exercise-induced angina (1 = Yes, 0 = No)                                    |
| `oldpeak`    | Float              | ST depression induced by exercise compared to rest                           |
| `slope`      | Categorical        | Slope of the peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping) |
| `ca`         | Integer (0-4)       | Number of major vessels colored by fluoroscopy                               |
| `thal`       | Categorical        | Thalassemia status (1: Normal, 2: Fixed Defect, 3: Reversible Defect)       |
| `target`     | Binary             | Target variable (1 = Heart disease present, 0 = No heart disease)            |

## System Requirements
### Software:
- **Python 3.7+**
- **IDE/Editor**: Visual Studio Code, Jupyter Notebook, or Google Colab
- **Libraries**:
  - `pandas` - Data manipulation
  - `numpy` - Numerical operations
  - `seaborn` & `matplotlib` - Plotting static charts
  - `plotly` - Interactive visualizations
  - `streamlit` - Web app framework
  - `sklearn` (optional, for later model integration)
  
### Hardware:
- **RAM**: Minimum 4GB (8GB preferred)
- **Processor**: Dual-core or higher
- **Browser**: Chrome/Firefox/Edge

### Optional Tools:
- `ngrok` - For exposing the local Streamlit app to the web (in Google Colab)
- Streamlit Cloud or Heroku for final deployment

## Implementation Details
### Step 1: Data Loading and Cleaning
- Used `pandas.read_csv()` to load the dataset.
- Checked for missing values using `df.isnull().sum()`.
- Encoded categorical labels (e.g., target labels) for clarity.

### Step 2: Exploratory Data Analysis (EDA)
- **Histograms**: To analyze the distribution of attributes like age, cholesterol, and blood pressure.
- **Boxplots**: To detect outliers.
- **Pie Charts**: For visualizing the proportion of categories like chest pain type.
- **Heatmap**: To identify correlations between numeric features.
- **Pairplots**: To visualize feature interactions.

### Step 3: Dashboard Development (Streamlit)
- Created an interactive sidebar with filters:
  - Filter data by age, sex, and chest pain type.
- Added tabs for different types of visualizations: demographics, symptoms, correlation.
- Used `plotly.express` for interactive visualizations such as:
  - Scatter matrix
  - Grouped bar charts
  - Dynamic pie charts

### Step 4: Visualizations
- **Pie Chart**: Heart disease distribution (patients with/without heart disease).
- **Histogram**: Age distribution highlighting high-risk age groups.
- **Bar Chart**: Frequency of chest pain types among patients with heart disease.
- **Scatter Plot**: Age vs. Maximum Heart Rate to study exertion responses.
- **Heatmap**: Correlation analysis (e.g., high correlation between `thalach`, `exang`, and `target`).

### Step 5: Deployment
- Locally tested with Streamlit:
  ```bash
  streamlit run main.py
