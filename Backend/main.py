from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="UPI Fraud Detection API", version="0.1.0")

model = joblib.load("C:/Users/Isha/Desktop/AI_ML/UPI_Fraud_Detection/Model/isolation_forest.pkl")
FEATURES = list(model.feature_names_in_)


class Transaction(BaseModel):
    amount_inr: float
    mcc: float
    latitude: float
    longitude: float
    distance_from_home_km: float
    device_age_hours: float
    is_new_device: int
    is_rooted_or_emulator: int
    sim_swap_last_7d: int
    hour_of_day: int
    day_of_month: int
    is_weekend: int
    is_night_txn: int
    usual_hour_start: int
    usual_hour_end: int
    is_unusual_hour: int
    sender_account_age_days: float
    hist_avg_amount: float
    hist_std_amount: float
    hist_avg_daily_txns: float
    balance_before_txn: float
    amount_to_balance_ratio: float
    amount_to_hist_avg_ratio: float
    amount_zscore: float
    txn_count_last_1h: int
    txn_count_last_24h: int
    amount_sum_last_24h: float
    mins_since_prev_txn: float
    km_from_prev_txn: float
    speed_kmph_from_prev: float
    receiver_account_age_days: float
    is_first_time_receiver: int
    same_receiver_txn_count_24h: int
    same_receiver_txn_count_7d: int
    receiver_unique_senders_7d: int
    sender_avg_amount: float
    amount_deviation: float
    amount_ratio: float
    user_transaction_count: int
    recipient_transaction_count: int
    device_change: int


class Prediction(BaseModel):
    anomaly: int
    risk_level: str
    alert: bool
    anomaly_score: float


@app.get("/")
def home():
    return {"message": "UPI Fraud Detection API is running"}


@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction):
    data = pd.DataFrame([transaction.model_dump()])[FEATURES]

    pred = model.predict(data)[0]
    score = model.decision_function(data)[0]
    is_anomaly = pred == -1

    return Prediction(
        anomaly=int(is_anomaly),
        risk_level="High Risk" if is_anomaly else "Normal",
        alert=bool(is_anomaly),
        anomaly_score=float(score),
    )