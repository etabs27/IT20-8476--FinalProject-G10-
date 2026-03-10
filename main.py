import joblib
import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress version mismatch warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Load the saved model, scaler, and label encoder
try:
    best_lr_model_smote = joblib.load('best_lr_model_smote.joblib')
    scaler = joblib.load('scaler.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    print("Model, scaler, and label encoder loaded successfully.")
except FileNotFoundError:
    print("Error: One or more necessary files (model, scaler, or label encoder) not found.")
    print("Please ensure 'best_lr_model_smote.joblib', 'scaler.joblib', and 'label_encoder.joblib' are in the same directory.")
    exit()

def get_user_input():
    """Collects feature inputs from the user."""
    print("\nEnter feature values for prediction:")
    try:
        product_category = input("Product Category (e.g., 'Beauty', 'Fashion', 'Electronics', 'Home & Kitchen', 'Books', 'Toys', 'Automotive', 'Sports'): ")
        product_price = float(input("Product Price: "))
        quantity = int(input("Quantity: "))
        delivery_days = int(input("Delivery Days: "))
        customer_rating = float(input("Customer Rating (1.0-5.0): "))
        order_month = int(input("Order Month (1-12): "))
        order_dayofweek = int(input("Order Day of Week (0=Monday, 6=Sunday): "))

        user_data = {
            'product_category': product_category,
            'product_price': product_price,
            'quantity': quantity,
            'delivery_days': delivery_days,
            'customer_rating': customer_rating,
            'order_month': order_month,
            'order_dayofweek': order_dayofweek
        }
        return user_data
    except ValueError:
        print("Invalid input. Please ensure numerical values are entered correctly.")
        exit()

def preprocess_input(user_data, label_encoder, scaler):
    """Preprocesses user input data for the model."""
    df_input = pd.DataFrame([user_data])

    # Encode 'product_category' and handle unseen categories
    try:
        df_input['product_category'] = label_encoder.transform(df_input['product_category'])
    except ValueError:
        print(f"Error: Product category '{user_data['product_category']}' is not recognized.")
        print("Please use one of the categories used during training.")
        exit()

    # Ensure column order matches training data
    feature_order = ['product_category', 'product_price', 'quantity', 'delivery_days',
                     'customer_rating', 'order_month', 'order_dayofweek']
    df_input = df_input[feature_order]

    # Scale numerical features
    X_input_scaled = scaler.transform(df_input)
    return X_input_scaled

def main():
    user_input_data = get_user_input()
    if user_input_data is None:  # Exit if input was invalid
        return

    processed_input = preprocess_input(user_input_data, label_encoder, scaler)

    # Make prediction
    prediction = best_lr_model_smote.predict(processed_input)
    prediction_proba = best_lr_model_smote.predict_proba(processed_input)

    print("\n--- Prediction Results ---")
    if prediction[0] == 1:
        print("The item is predicted to be RETURNED.")
    else:
        print("The item is predicted to be NOT RETURNED.")
    print(f"Probability of NOT RETURNED: {prediction_proba[0][0]:.4f}")
    print(f"Probability of RETURNED: {prediction_proba[0][1]:.4f}")

    # Optimal threshold from previous analysis
    optimal_threshold = 0.4917

    if prediction_proba[0][1] >= optimal_threshold:
        print(f"(Based on the previously found optimal threshold of {optimal_threshold:.4f}, this item would also be predicted as RETURNED.)")
    else:
        print(f"(Based on the previously found optimal threshold of {optimal_threshold:.4f}, this item would also be predicted as NOT RETURNED.)")

if __name__ == '__main__':
    main()