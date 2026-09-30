import streamlit as st
import pandas as pd
import joblib

# Load dataset
data = pd.read_csv("smartphone_dataset_1M.csv")

features = ['ram_gb', 'storage_gb', 'battery_mah', 'display_size_inch']

X = data[features].dropna()
data = data.loc[X.index]

# Load model
knn = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📱 AI Smartphone Recommendation System")

ram = st.selectbox("RAM (GB)", [2,4,6,8,12,16], index=3)
storage = st.selectbox("Storage (GB)", [32,64,128,256,512], index=3)
battery = st.number_input("Battery (mAh)", 3000, 10000, 6000)
display = st.number_input("Display Size (inch)", 5.0, 8.0, 6.7)

if st.button("Recommend Phones"):

    user_input = [[ram, storage, battery, display]]

    user_scaled = scaler.transform(user_input)

    distances, indices = knn.kneighbors(user_scaled)

    result = data.iloc[indices[0]][[
        'brand',
        'model_name',
        'price_inr',
        'ram_gb',
        'storage_gb',
        'battery_mah',
        'display_size_inch'
    ]]

    st.subheader("Top Recommended Phones")

    st.dataframe(result.reset_index(drop=True))
    import streamlit as st

st.title("Test App")

st.write("If you can see this text, Streamlit is working!")