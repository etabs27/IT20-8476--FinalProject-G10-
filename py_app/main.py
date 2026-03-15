import joblib
import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress version mismatch warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Load the saved model and metadata
try:
    artifact = joblib.load('lgbm_model_and_params(2).joblib')
    if isinstance(artifact, dict) and 'model' in artifact:
        model = artifact['model']
        optimal_threshold = artifact.get('optimal_threshold', 0.4917)
    else:
        model = artifact
        optimal_threshold = 0.4917
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file not found.")
    print("Please ensure 'lgbm_model_and_params(2).joblib' is in the same directory.")
    exit()

def get_user_input():
    """Collects feature inputs from the user."""
    print("\nPlease enter the following details for the product:")
    try:
        product_category = input("Enter product_category (options: Beauty, Fashion, Electronics, Sports, Home, Toys): ")
        product_price = float(input("Enter product_price (any positive number): "))
        quantity = int(input("Enter quantity (any positive integer): "))
        order_date = input("Enter order_date (YYYY-MM-DD): ")
        region = input("Enter region (Europe, North America, Oceania, Africa, Asia, South America): ")
        payment_method = input("Enter payment_method (BankTransfer, CreditCard, Cash, PayPal): ")
        delivery_days = int(input("Enter delivery_days (any positive integer): "))
        customer_rating = float(input("Enter customer_rating (1.0-5.0): "))
        discount_percent = float(input("Enter discount_percent (0-100): "))
        revenue = float(input("Enter revenue (any positive number): "))

        order_date_parsed = pd.to_datetime(order_date, format='%Y-%m-%d', errors='coerce')
        if pd.isna(order_date_parsed):
            raise ValueError("order_date must be in YYYY-MM-DD format")

        return {
            'product_category': product_category,
            'product_price': product_price,
            'quantity': quantity,
            'order_date': order_date_parsed,
            'region': region,
            'payment_method': payment_method,
            'delivery_days': delivery_days,
            'customer_rating': customer_rating,
            'discount_percent': discount_percent,
            'revenue': revenue
        }
    except ValueError as e:
        print(f"Invalid input: {e}")
        exit()

def build_feature_vector(user_data, artifact):
    feature_cols = artifact.get('feature_columns', [])
    revenue = float(user_data.get('revenue', float(user_data['product_price']) * float(user_data['quantity'])))
    discount_percent = float(user_data.get('discount_percent', 0.0))
    order_value = float(user_data['product_price']) * float(user_data['quantity'])
    discount_amount = order_value * (discount_percent / 100.0)
    net_revenue = order_value - discount_amount
    price_per_delivery_day = float(user_data['product_price']) / max(1.0, float(user_data['delivery_days']))

    vector = []
    for col in feature_cols:
        if col == 'product_price':
            vector.append(float(user_data['product_price']))
        elif col == 'quantity':
            vector.append(float(user_data['quantity']))
        elif col == 'delivery_days':
            vector.append(float(user_data['delivery_days']))
        elif col == 'customer_rating':
            vector.append(float(user_data['customer_rating']))
        elif col == 'discount_percent':
            vector.append(discount_percent)
        elif col == 'revenue':
            vector.append(revenue)
        elif col == 'order_month':
            vector.append(float(user_data['order_date'].month))
        elif col == 'order_day':
            vector.append(float(user_data['order_date'].day))
        elif col == 'order_weekday':
            vector.append(float(user_data['order_date'].dayofweek))
        elif col == 'order_value':
            vector.append(order_value)
        elif col == 'discount_amount':
            vector.append(discount_amount)
        elif col == 'net_revenue':
            vector.append(net_revenue)
        elif col == 'price_per_delivery_day':
            vector.append(price_per_delivery_day)
        elif col.startswith('product_category_'):
            vector.append(1.0 if col == f"product_category_{user_data['product_category']}" else 0.0)
        elif col.startswith('region_'):
            region_clean = str(user_data.get('region', 'North America')).replace(' ', '_')
            vector.append(1.0 if col == f"region_{region_clean}" else 0.0)
        elif col.startswith('payment_method_'):
            payment_clean = str(user_data.get('payment_method', 'CreditCard')).replace(' ', '')
            vector.append(1.0 if col == f"payment_method_{payment_clean}" else 0.0)
        else:
            vector.append(0.0)
    return pd.DataFrame([vector], columns=feature_cols)



def main():
    user_input_data = get_user_input()
    if user_input_data is None:  # Exit if input was invalid
        return

    processed_input = build_feature_vector(user_input_data, artifact)

    # Make prediction
    prediction_proba = model.predict_proba(processed_input)[0]
    prediction = int(prediction_proba[1] >= optimal_threshold)

    print("\n--- Prediction Results ---")
    if prediction == 1:
        print("The item is predicted to be RETURNED.")
    else:
        print("The item is predicted to be NOT RETURNED.")
    print(f"Probability of NOT RETURNED: {prediction_proba[0]:.4f}")
    print(f"Probability of RETURNED: {prediction_proba[1]:.4f}")

    print(f"(Based on the previously found optimal threshold of {optimal_threshold:.4f}, this item is predicted as {'RETURNED' if prediction == 1 else 'NOT RETURNED'}.)")

if __name__ == '__main__':
    main()