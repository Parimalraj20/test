"""
Linear Regression - Simple Example
Author: Sample
Description:
    - Generates sample data
    - Trains a Linear Regression model
    - Evaluates the model
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def generate_data():
    """
    Generate sample dataset
    """
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10      # Feature
    y = 2.5 * X + 5 + np.random.randn(100, 1)  # Target with noise

    return X, y


def train_model(X, y):
    """
    Train Linear Regression model
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("📊 Model Evaluation")
    print("MSE:", mse)
    print("R2 Score:", r2)


def main():
    # Generate data
    X, y = generate_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = train_model(X_train, y_train)

    # Print model parameters
    print("📈 Model Parameters")
    print("Coefficient:", model.coef_[0][0])
    print("Intercept:", model.intercept_[0])

    # Evaluate
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()