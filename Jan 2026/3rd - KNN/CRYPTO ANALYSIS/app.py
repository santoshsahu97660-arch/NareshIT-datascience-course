import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# -------------------------------
# Load trained model
# -------------------------------
model_path = 'random_forest_model.pkl'
with open(model_path, 'rb') as file:
    model_rf = pickle.load(file)

# -------------------------------
# Prediction function
# -------------------------------
def predict_btc_price(input_data):
    prediction = model_rf.predict(input_data)
    return float(prediction[0])

# -------------------------------
# Main App
# -------------------------------
def main():
    st.set_page_config(page_title="BTC Price Prediction", layout="centered")
    st.title("🚀 BTC Close Price Prediction & Trade Signal")

    st.sidebar.header("🔢 Input Market Data")

    # User Inputs
    current_btc_price = st.sidebar.number_input(
        "Current BTC Price (USD)", min_value=0.0, format="%.2f"
    )

    usdt_close = st.sidebar.number_input(
        "USDT Close Price", min_value=0.0, format="%.2f"
    )
    usdt_volume = st.sidebar.number_input(
        "USDT Volume", min_value=0.0, format="%.2f"
    )
    bnb_close = st.sidebar.number_input(
        "BNB Close Price", min_value=0.0, format="%.2f"
    )
    bnb_volume = st.sidebar.number_input(
        "BNB Volume", min_value=0.0, format="%.2f"
    )

    # Input DataFrame
    input_data = pd.DataFrame({
        'USDT_Close': [usdt_close],
        'USDT_Volume': [usdt_volume],
        'BNB_Close': [bnb_close],
        'BNB_Volume': [bnb_volume]
    })

    # -------------------------------
    # Prediction Button
    # -------------------------------
    if st.button("📊 Predict BTC Price"):
        predicted_price = predict_btc_price(input_data)

        st.subheader("📈 Prediction Result")
        st.write(f"**Predicted BTC Close Price:** 💲 {predicted_price:,.2f}")

        # -------------------------------
        # Buy / Sell Logic
        # -------------------------------
        diff = predicted_price - current_btc_price

        if diff > 0:
            signal = "🟢 BUY"
            st.success("📈 Signal: BUY (Price Expected to Rise)")
        elif diff < 0:
            signal = "🔴 SELL"
            st.error("📉 Signal: SELL (Price Expected to Fall)")
        else:
            signal = "🟡 HOLD"
            st.warning("⏸️ Signal: HOLD")

        # -------------------------------
        # Graph
        # -------------------------------
        st.subheader("📊 Price Comparison Graph")

        price_df = pd.DataFrame({
            "Type": ["Current BTC Price", "Predicted BTC Price"],
            "Price": [current_btc_price, predicted_price]
        })

        fig, ax = plt.subplots()
        ax.bar(price_df["Type"], price_df["Price"])
        ax.set_ylabel("Price (USD)")
        ax.set_title("BTC Price Comparison")

        st.pyplot(fig)

        # -------------------------------
        # Summary Box
        # -------------------------------
        st.info(f"""
        **Trading Summary**
        - Current BTC Price: 💲 {current_btc_price:,.2f}
        - Predicted BTC Price: 💲 {predicted_price:,.2f}
        - Recommendation: **{signal}**
        """)

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    main()
