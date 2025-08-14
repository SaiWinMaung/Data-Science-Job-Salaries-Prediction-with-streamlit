import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
h1, h2 = st.columns([0.2, 0.8])
with h1:
    st.image("sai_logo.jpg")
with h2:
    st.title("Data Science Professional Job Salaries Prediction")
df = pd.read_csv("salaries.csv")
df["experience_level"].replace(
    {"SE": "Senior", "EX": "Executive", "EN": "Entry-level", "MI": "Mid-level"},
    inplace=True,
)
df["employment_type"] = df["employment_type"].replace(
    {"FT": "Full-time", "PT": "Part-time", "CT": "Contract", "FL": "Freelance"}
)
df["company_size"] = df["company_size"].replace(
    {"M": "Medium", "L": "Large", "S": "Small"}
)

le = LabelEncoder()
df["experience_level_encode"] = le.fit_transform(df["experience_level"])

df["employment_type_encode"] = le.fit_transform(df["employment_type"])

df["company_size_encode"] = le.fit_transform(df["company_size"])

df["job_title_encode"] = le.fit_transform(df["job_title"])

scaler = StandardScaler()
df["work_year_scaled"] = scaler.fit_transform(df[["work_year"]])

X = df[
    [
        "work_year_scaled",
        "experience_level_encode",
        "employment_type_encode",
        "job_title_encode",
        "company_size_encode",
    ]
]

y = df["salary_in_usd"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Model = LinearRegression()
Model.fit(X_train, y_train)

with st.sidebar:

    st.markdown("## Please Select Options")

    year = st.selectbox("Years", df["work_year"].unique())

    experience = st.radio("Experience Level", df["experience_level"].unique())

    employment = st.radio("Employment Type", df["employment_type"].unique())

    job = st.selectbox("Job Title", df["job_title"].unique())

    company = st.radio("Company Size", df["company_size"].unique())

    work_year_scaled = scaler.transform([[year]])[0][0]

    if experience == "Executive":
        experience_level_encode = 3

    elif experience == "Senior":
        experience_level_encode = 2

    elif experience == "Mid-level":
        experience_level_encode = 1
    else:
        experience_level_encode = 0

    if employment == "Full-time":
        employment_type_encode = 3

    elif employment == "Part-time":
        employment_type_encode = 2

    elif employment == "Contract":
        employment_type_encode = 1

    else:
        employment_type_encode = 0

    job_title_encode = le.transform([job])[0]

    if company == "Large":
        company_size_encode = 2

    elif company == "Medium":
        company_size_encode = 1

    else:
        company_size_encode = 0

input_data = np.array(
    [
        [
            work_year_scaled,
            experience_level_encode,
            employment_type_encode,
            job_title_encode,
            company_size_encode,
        ]
    ]
)
predicted_salary = round(Model.predict(input_data)[0], 3)
st.header("Please input features to predict salary")
st.markdown(
    f"<span style='font-size:24px;'>Predicted Salary : $ </span>"
    f"<span style='color:green; font-size:24px; font-weight:bold;'>{predicted_salary}</span>",
    unsafe_allow_html=True
)