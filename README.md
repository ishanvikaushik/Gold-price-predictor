# 📈 Gold Price Predictor (India)

Predict the future price of gold in India (₹/gram) for any date up to 2 years ahead using historical data and Meta’s Prophet time series model. This app is built with [Streamlit](https://streamlit.io/) for an interactive web experience.

## Features

- **Data Quality Check:** Ensures clean and valid input data.
- **Forecasting:** Predicts gold prices for up to 2 years into the future.
- **Date Picker:** Select any future date to get a price prediction and confidence interval.
- **Interactive Charts:** Visualize past and forecasted trends.
- **Model Evaluation:** Displays RMSE for model accuracy.
- **Recent Predictions:** Shows predicted prices for the next 7 days.

## How to Run

1. **Clone this repository:**
    ```bash
    git clone https://github.com/ishnvikaushik/Gold-price-predictor.git
    cd gold-price-predictor
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Add data:**
    - Place your cleaned gold price data as `data/gold_price_india_cleaned.csv` (with columns: `Date`, `Price`).

4. **Run the app:**
    ```bash
    streamlit run streamlit_app.py
    ```

## File Structure

- `streamlit_app.py` — Main Streamlit application.
- `data/gold_price_india_cleaned.csv` — Historical gold price data (not included).
- `requirements.txt` — Python dependencies.

## Dependencies

- streamlit
- pandas
- prophet
- plotly
- scikit-learn
- numpy


---

**Note:** This project is for educational purposes. Predictions are based on historical trends and may not reflect actual future prices.
