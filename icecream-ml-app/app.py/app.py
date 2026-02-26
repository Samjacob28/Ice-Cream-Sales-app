import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ==============================
# Page Title
# ==============================
st.title("🍦 Ice Cream Sales Prediction App")

# ==============================
# Load Dataset
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("Icecream Sales wr Rain and Temperature.csv")
    df["Did it rain on that day?"] = df["Did it rain on that day?"].map({"Yes": 1, "No": 0})
    return df

df = load_data()

# ==============================
# Train Model
# ==============================
X = df.drop("Ice Cream Sales ($,thousands)", axis=1)
y = df["Ice Cream Sales ($,thousands)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Model Accuracy
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)

st.write("### 📊 Model Accuracy (R² Score):", round(accuracy, 3))

# ==============================
# User Inputs
# ==============================
st.sidebar.header("Enter Day Details")

temperature = st.sidebar.slider("Temperature (F)", 50, 120, 85)
price = st.sidebar.slider("Ice-cream Price ($)", 1, 10, 2)
tourists = st.sidebar.slider("Number of Tourists (thousands)", 0, 200, 80)
rain = st.sidebar.selectbox("Did it Rain?", ["No", "Yes"])

rain_value = 1 if rain == "Yes" else 0

# ==============================
# Prediction
# ==============================
input_data = pd.DataFrame({
    "Temperature (F)": [temperature],
    "Ice-cream Price ($)": [price],
    "Number of Tourists (thousands)": [tourists],
    "Did it rain on that day?": [rain_value]
})

prediction = model.predict(input_data)

st.subheader("💰 Predicted Ice Cream Sales:")
st.success(f"${prediction[0]:.2f} thousand")

# ==============================
# Show Raw Data (Optional)
# ==============================
if st.checkbox("Show Dataset"):
    st.write(df)
