import streamlit as st
import pandas as pd
import pickle
import requests

# ---------------- LOAD MODELS ---------------- #
price_model = pickle.load(open("price_model.pkl", "rb"))
le_crop = pickle.load(open("le_crop.pkl", "rb"))
le_state = pickle.load(open("le_state.pkl", "rb"))
le_district = pickle.load(open("le_district.pkl", "rb"))

# ---------------- LOAD DATASET ---------------- #
price_df = pd.read_csv("crop_price.csv", encoding="latin1")

# ---------------- WEATHER API ---------------- #
API_KEY = "9886af1c13e118d487e9d080f995fde6"   # 🔴 Put your OpenWeather API key

def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        res = requests.get(url).json()
        
        if res.get("cod") != 200:
            return None

        return {
            "temperature": res["main"]["temp"],
            "humidity": res["main"]["humidity"]
        }
    except:
        return None

# ---------------- SAFE ENCODING ---------------- #
def safe_transform(le, value):
    value = value.lower()
    for i, v in enumerate(le.classes_):
        if v.lower() == value:
            return i
    return 0

# ---------------- CURRENT PRICE (FIXED) ---------------- #
def get_current_price(crop):
    try:
        # correct column names from your dataset
        data = price_df[price_df["commodity"].str.lower() == crop.lower()]
        
        if len(data) == 0:
            return 1500
        
        return data["modal_price"].mean()
    
    except:
        return 1500

# ---------------- HARVEST MONTH ---------------- #
def get_harvest_month(crop):
    crop_map = {
        "rice": 10,
        "wheat": 4,
        "maize": 9,
        "cotton": 11,
        "jute": 8,
        "sugarcane": 12
    }
    return crop_map.get(crop, 6)

# ---------------- MAIN LOGIC ---------------- #
def recommend(state, district, city):

    weather = get_weather(city)

    if weather is None:
        return None

    temp = weather["temperature"]
    humidity = weather["humidity"]
    rainfall = 100  # assumed

    state_enc = safe_transform(le_state, state)
    district_enc = safe_transform(le_district, district)

    results = []

    for crop in le_crop.classes_:

        crop_enc = safe_transform(le_crop, crop)
        month = get_harvest_month(crop)

        future_price = price_model.predict([[
            crop_enc,
            state_enc,
            district_enc,
            temp,
            humidity,
            rainfall,
            month
        ]])[0]

        current_price = get_current_price(crop)
        profit = future_price - current_price

        results.append({
            "Crop": crop,
            "Current Price": current_price,
            "Future Price": future_price,
            "Profit": profit
        })

    return sorted(results, key=lambda x: x["Profit"], reverse=True)[:5]

# ---------------- STREAMLIT UI ---------------- #
st.set_page_config(page_title="Crop Predictor", layout="centered")

st.title("AgroMind")

st.write("Get the best crops to grow based on location and weather")

state = st.selectbox("Select State", le_state.classes_)
district = st.selectbox("Select District", le_district.classes_)
city = st.text_input("Enter City (for Weather Data)")

if st.button("Get Recommendations"):

    if city.strip() == "":
        st.warning(" Please enter a city name")

    else:
        results = recommend(state, district, city)

        if results is None:
            st.error(" Invalid city or weather API issue")
        else:
            st.subheader(" Top 5 Profitable Crops")

            for r in results:
                st.markdown(f"""
                ###  {r['Crop'].title()}
                 Current Price: ₹{r['Current Price']:.2f} per quintal  
                 Future Price: ₹{r['Future Price']:.2f} per quintal  
                 Expected Profit: ₹{r['Profit']:.2f}  
                ---
                """)