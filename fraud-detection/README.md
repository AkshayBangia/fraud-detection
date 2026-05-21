# 🛡️ FraudGuard — Credit Card Fraud Detection

A portfolio ML project with a full Streamlit web app for detecting credit card fraud using XGBoost.
Link after Deployment- https://fraud-detection-edsmcszxak6ghbi4degclh.streamlit.app/

---

## 📁 Project Structure

```
fraud-detection/
├── app.py                  ← Streamlit web app (run this!)
├── requirements.txt
├── data/
│   └── creditcard.csv      ← Download from Kaggle (see below)
├── models/
│   ├── fraud_model.pkl     ← Auto-generated after training
│   └── scaler.pkl          ← Auto-generated after training
└── src/
    ├── eda.py              ← Exploratory data analysis
    └── train_model.py      ← Model training script
```

---

## 🚀 Setup — Step by Step

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download `creditcard.csv` and place it inside the `data/` folder:
```
data/creditcard.csv
```

### 3. (Optional) Run EDA
Explore the data and generate plots:
```bash
python src/eda.py
```

### 4. Train the model
```bash
python src/train_model.py
```
This trains Logistic Regression, Random Forest, and XGBoost — and saves the best model.

### 5. Launch the web app
```bash
streamlit run app.py
```
Open your browser at http://localhost:8501

---

## 🌐 Deploy for Free (Streamlit Cloud)

1. Push this project to a GitHub repo
2. Go to https://streamlit.io/cloud
3. Click "New app" → select your repo → set `app.py` as the main file
4. Click Deploy!

> Note: For cloud deployment, you'll need to commit `creditcard.csv` (it's 150MB — consider using Git LFS) or load it from a URL.

---

## 📊 Results

| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|----|---------|
| Logistic Regression | 0.863 | 0.871 | 0.867 | 0.974 |
| Random Forest | 0.924 | 0.901 | 0.912 | 0.979 |
| **XGBoost** ✅ | **0.942** | **0.918** | **0.930** | **0.983** |

---

## 💡 Key Learnings

- **Class imbalance** is the core challenge — only 0.17% of transactions are fraud
- **Accuracy is misleading** — always use Precision, Recall, F1, and ROC-AUC
- `scale_pos_weight` in XGBoost is a simple and effective way to handle imbalance
- Fraud transactions tend to have lower amounts and occur at unusual hours

---

*Built with Python · scikit-learn · XGBoost · Streamlit*
