import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("india_gdp_1960_2025.csv")

# Train model
X = df[['Year']]
y = df['GDP (Billions USD)']
model = LinearRegression()
model.fit(X, y)
df['Predicted GDP'] = model.predict(X)

# Streamlit UI
st.title("🇮🇳 India GDP Prediction App (1960 - 2025)")
st.markdown("Built with Streamlit + Machine Learning")

# Line chart
st.subheader("📈 Historical vs Predicted GDP")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df['Year'], df['GDP (Billions USD)'], label='Actual GDP')
ax.plot(df['Year'], df['Predicted GDP'], label='Predicted GDP')
ax.set_xlabel("Year")
ax.set_ylabel("GDP (Billions USD)")
ax.set_title("India GDP Over the Years")
ax.legend()
st.pyplot(fig)

# Future prediction
st.subheader("🔮 Predict Future GDP")
future_year = st.number_input("Enter a future year (2026+):", min_value=2026, max_value=2100, value=2026)
future_df = pd.DataFrame({'Year': [future_year]})
future_pred = model.predict(future_df)
st.success(f"Predicted GDP for {future_year}: **${future_pred[0]:,.2f} Billion USD**")
