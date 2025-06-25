import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data():
    """Load and preprocess the housing data"""
    # Load dataset
    data = pd.read_csv('data/HousingData.csv')
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    return data_imputed

def train_model(X_train, y_train):
    """Train and tune the Gradient Boosting model"""
    # Initialize model
    gb = GradientBoostingRegressor(random_state=42)
    
    # Define parameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Perform grid search
    grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Evaluation:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Plot feature importance
    feature_importance = model.feature_importances_
    features = X_train.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.close()

def save_artifacts(model, scaler):
    """Save model and scaler"""
    joblib.dump(model, 'models/boston_housing_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Model and scaler saved successfully")

if __name__ == "__main__":
    # Load and preprocess data
    data = load_and_preprocess_data()
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training model...")
    best_model, best_params = train_model(X_train_scaled, y_train)
    print(f"Best parameters found: {best_params}")
    
    # Evaluate
    evaluate_model(best_model, X_test_scaled, y_test)
    
    # Save artifacts
    save_artifacts(best_model, scaler)