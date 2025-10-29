import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="🏡 House Price Prediction", layout="centered")

def load_model(filename):
    try:
        if os.path.exists(filename):
            return joblib.load(filename)
        else:
            st.warning(f"⚠️ {filename} not found.")
            return None
    except Exception as e:
        st.error(f"❌ Could not load {filename}: {e}")
        return None

lin_reg = load_model("Linear_Regression.pkl")
rf_model = load_model("Random_Forest.pkl")
gb_model = load_model("Gradient_Boosting.pkl")

st.title("🏡 House Price Prediction System")
st.markdown("### Compare predictions from **Linear Regression, Random Forest & Gradient Boosting**")

st.sidebar.header("Enter House Details")
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 2)
sqft_living = st.sidebar.number_input("Living Area (sq ft)", 300, 10000, 1800)
sqft_lot = st.sidebar.number_input("Lot Area (sq ft)", 500, 50000, 5000)
floors = st.sidebar.slider("Floors", 1, 3, 1)
waterfront = st.sidebar.selectbox("Waterfront", [0, 1])
view = st.sidebar.slider("View", 0, 4, 0)
condition = st.sidebar.slider("Condition (1=Poor, 5=Excellent)", 1, 5, 3)
sqft_above = st.sidebar.number_input("Sqft Above", 300, 10000, 1500)
sqft_basement = st.sidebar.number_input("Sqft Basement", 0, 5000, 0)
house_age = st.sidebar.number_input("House Age (years)", 0, 150, 30)
years_since_renovation = st.sidebar.number_input("Years Since Renovation", 0, 100, 10)
city_encoded = st.sidebar.number_input("City (Encoded)", 0, 100, 10)
statezip_encoded = st.sidebar.number_input("StateZip (Encoded)", 0, 50, 5)

features = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view,
                      condition, sqft_above, sqft_basement, house_age, years_since_renovation,
                      city_encoded, statezip_encoded]])

st.subheader("🔮 Predictions")
preds = {}
if lin_reg:
    preds["Linear Regression"] = lin_reg.predict(features)[0]
    st.write(f"**Linear Regression:** ${preds['Linear Regression']:,.2f}")

if rf_model:
    preds["Random Forest"] = rf_model.predict(features)[0]
    st.write(f"**Random Forest:** ${preds['Random Forest']:,.2f}")

if gb_model:
    preds["Gradient Boosting"] = gb_model.predict(features)[0]
    st.write(f"**Gradient Boosting:** ${preds['Gradient Boosting']:,.2f}")

if not preds:
    st.error("❌ No models available for prediction. Please check your .pkl files.")

if preds:
    fig, ax = plt.subplots()
    ax.bar(preds.keys(), preds.values(), color=["blue", "green", "orange"][:len(preds)])
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title("Model Predictions Comparison")
    st.pyplot(fig)

st.markdown("---")
st.markdown("👨‍💻 **Developed for Internship Task – House Price Prediction**")