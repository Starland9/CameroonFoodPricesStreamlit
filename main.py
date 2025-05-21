import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

DATA_URL = "https://data.humdata.org/dataset/f6ed9f06-36b9-43b2-97f3-fc0b8e16d155/resource/9deec824-1e42-4a91-b704-b73fd250ec5a/download/wfp_food_prices_cmr.csv"


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    data = data.iloc[1:]
    data = data.rename(
        columns={'admin1': 'region', 'admin2': 'department', 'market': 'city', 'latitude': 'lat', 'longitude': 'lon'})
    data = data.dropna()

    # transform lon and lat to float
    data['lat'] = data['lat'].astype(float)
    data['lon'] = data['lon'].astype(float)
    data['price'] = data['price'].astype(float)
    data['usdprice'] = data['usdprice'].astype(float)
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
    return data


data = load_data()

st.header('Food Prices in Cameroon')
st.write(data)

st.subheader("Description of datas")
st.write(data.describe())

st.subheader('Map of Food Prices')
st.map(data)

st.header("Graphical details")
selected_category = st.selectbox(
    'Select a category',
    data['category'].unique()
)

selected_commodity = st.selectbox(
    'Select a commodity',
    data[data['category'] == selected_category]['commodity'].unique()
)

filtered_data = data[(data['category'] == selected_category) & (data['commodity'] == selected_commodity)]

left, right = st.columns(2)
with left:
    st.subheader("Price vs. Date")
    st.line_chart(filtered_data.groupby('date')['price'].mean(), x_label='date', y_label='price(XAF)')

with right:
    st.subheader("Price vs. Region")
    st.bar_chart(filtered_data.groupby('region')['price'].mean(), x_label='Region', y_label='price(XAF)')


@st.cache_data
def predict_price(region, department, city, category, commodity):
    # Créer une copie du jeu de données
    transformed_data = data.copy()
    
    # Créer les encoders pour chaque catégorie
    region_encoder = pd.factorize(data['region'])[1]
    department_encoder = pd.factorize(data['department'])[1]
    city_encoder = pd.factorize(data['city'])[1]
    category_encoder = pd.factorize(data['category'])[1]
    commodity_encoder = pd.factorize(data['commodity'])[1]
    
    # Transformer les données d'entraînement
    transformed_data['region'] = pd.Categorical(transformed_data['region'], categories=region_encoder).codes
    transformed_data['department'] = pd.Categorical(transformed_data['department'], categories=department_encoder).codes
    transformed_data['city'] = pd.Categorical(transformed_data['city'], categories=city_encoder).codes
    transformed_data['category'] = pd.Categorical(transformed_data['category'], categories=category_encoder).codes
    transformed_data['commodity'] = pd.Categorical(transformed_data['commodity'], categories=commodity_encoder).codes
    
    # Préparer les données pour l'entraînement
    X = transformed_data[['region', 'department', 'city', 'category', 'commodity']]
    y = transformed_data['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Entraîner le modèle
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Préparer les données de prédiction
    prediction_data = pd.DataFrame({
        'region': [region],
        'department': [department],
        'city': [city],
        'category': [category],
        'commodity': [commodity]
    })
    
    # Transformer les données de prédiction avec les mêmes encoders
    prediction_data['region'] = pd.Categorical(prediction_data['region'], categories=region_encoder).codes
    prediction_data['department'] = pd.Categorical(prediction_data['department'], categories=department_encoder).codes
    prediction_data['city'] = pd.Categorical(prediction_data['city'], categories=city_encoder).codes
    prediction_data['category'] = pd.Categorical(prediction_data['category'], categories=category_encoder).codes
    prediction_data['commodity'] = pd.Categorical(prediction_data['commodity'], categories=commodity_encoder).codes

    # Faire la prédiction
    prediction = model.predict(prediction_data)[0]
    return prediction


st.title("Price Prediction")
# Créer les selectbox avec les valeurs uniques
region = st.selectbox("Region", data['region'].unique())
department = st.selectbox("Department", data[data['region'] == region]['department'].unique())
city = st.selectbox("City", data[data['department'] == department]['city'].unique())
category = st.selectbox("Category", data['category'].unique())
commodity = st.selectbox("Commodity", data[data['category'] == category]['commodity'].unique())

if st.button("Predict"):
    prediction = predict_price(region, department, city, category, commodity)
    st.write(f"Predicted price: {prediction} XAF")
