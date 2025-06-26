ShadowFox
# Boston House Price Prediction

This project builds a regression model to predict house prices in Boston based on features like number of rooms, crime rates, and more.

## 📂 Project Structure
```
boston-house-price-prediction/
├── data/                  # Contains HousingData.csv
├── models/                # Trained model (Linear Regression)
├── notebooks/             #contains analysis.
├── requirements.txt       # Python dependencies
├── README.md              # Project overview and instructions
```

## 🧠 Model Used
- Linear Regression (baseline)
- Mean Squared Error (MSE): 25.02
- R² Score: 0.659

## 🔧 How to Use
1. Install dependencies:
```
pip install -r requirements.txt
```

2. Load and use the model:
```python
import joblib
model = joblib.load('models/linear_regression_model.pkl')
predictions = model.predict(X_test)
```

## 📈 Dataset Features
Includes:
- CRIM: Crime rate
- RM: Number of rooms
- AGE, DIS, TAX, LSTAT, etc.

Target:
- MEDV: Median value of owner-occupied homes

---

Made with ❤️ for machine learning practice.

