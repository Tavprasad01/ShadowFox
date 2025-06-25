import joblib
import pandas as pd
import numpy as np

def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('models/boston_housing_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, None

def predict_price(input_data, model, scaler):
    """Make prediction on new data"""
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        return prediction[0]
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def get_user_input():
    """Get input features from user"""
    print("Enter the following features to predict Boston house price:")
    features = {
        'CRIM': float(input("Per capita crime rate (CRIM): ")),
        'ZN': float(input("Proportion of residential land (ZN): ")),
        'INDUS': float(input("Proportion of non-retail business acres (INDUS): ")),
        'CHAS': int(input("Charles River dummy variable (0 or 1, CHAS): ")),
        'NOX': float(input("Nitric oxides concentration (NOX): ")),
        'RM': float(input("Average number of rooms (RM): ")),
        'AGE': float(input("Proportion of old units (AGE): ")),
        'DIS': float(input("Weighted distances to employment centers (DIS): ")),
        'RAD': int(input("Index of accessibility to highways (RAD): ")),
        'TAX': float(input("Property tax rate (TAX): ")),
        'PTRATIO': float(input("Pupil-teacher ratio (PTRATIO): ")),
        'B': float(input("1000(Bk - 0.63)² (B): ")),
        'LSTAT': float(input("% lower status population (LSTAT): "))
    }
    return features

if __name__ == "__main__":
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    if model and scaler:
        # Get user input
        input_data = get_user_input()
        
        # Make prediction
        predicted_price = predict_price(input_data, model, scaler)
        
        if predicted_price is not None:
            print(f"\nPredicted Median House Price: ${predicted_price * 1000:,.2f}")
            print("(Note: Prices are in 1970s dollars)")
    else:
        print("Failed to load model. Please train the model first using train.py")
