import joblib
import pandas as pd
import numpy as np


def preprocess_input_data(user_input_data, feature_columns):
    """
    Preprocesses a single row of user input data to match the format
    expected by the trained LightGBM model.

    Args:
        user_input_data (dict): A dictionary containing raw user inputs for prediction.
                                 Expected keys: 'product_category', 'product_price', 'quantity',
                                 'order_date', 'region', 'payment_method', 'delivery_days',
                                 'customer_rating', 'discount_percent', 'revenue'.
        feature_columns (pd.Index): A pandas Index object containing the names of the columns
                                  the model was trained on, in the correct order.

    Returns:
        pd.DataFrame: A single-row DataFrame with processed and encoded features,
                      aligned with the model's training feature columns.
    """
    # Convert user input to a pandas DataFrame (single row)
    input_df = pd.DataFrame([user_input_data])

    # 1. Convert 'order_date' to datetime objects and extract features
    input_df['order_date'] = pd.to_datetime(input_df['order_date'])
    input_df['order_month'] = input_df['order_date'].dt.month
    input_df['order_day'] = input_df['order_date'].dt.day
    input_df['order_weekday'] = input_df['order_date'].dt.weekday  # Monday=0, Sunday=6

    # 2. Drop the original 'order_date' column
    input_df = input_df.drop(columns=['order_date'])

    # 3. Create new feature 'order_value'
    input_df['order_value'] = input_df['product_price'] * input_df['quantity']

    # 4. Create new feature 'discount_amount'
    input_df['discount_amount'] = input_df['order_value'] * (input_df['discount_percent'] / 100)

    # 5. Create new feature 'net_revenue'
    input_df['net_revenue'] = input_df['order_value'] - input_df['discount_amount']

    # 6. Create new feature 'price_per_delivery_day'
    # Handle potential division by zero if delivery_days can be 0. (Assuming min=1 from EDA)
    input_df['price_per_delivery_day'] = input_df['product_price'] / input_df['delivery_days']

    # Define categorical columns as they were in training
    categorical_cols_original = ['product_category', 'region', 'payment_method']

    # Apply one-hot encoding
    # Use get_dummies with parameters consistent with training (drop_first=True)
    input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols_original, drop_first=True)

    # Align columns with the training data's feature columns
    # Create a Series with all expected feature columns, initialized to 0
    processed_series = pd.Series(0, index=feature_columns, dtype=float)

    # Update the Series with values from the encoded input DataFrame
    for col in input_df_encoded.columns:
        if col in feature_columns:
            processed_series[col] = input_df_encoded[col].iloc[0]

    # Convert the Series to a DataFrame, ensuring the correct structure for prediction
    final_input_df = pd.DataFrame([processed_series])

    # Ensure boolean columns are converted to the correct numeric type (int) if necessary
    for col in final_input_df.columns:
        if final_input_df[col].dtype == 'bool':
            final_input_df[col] = final_input_df[col].astype(int)

    # Select and order columns according to the model's feature_columns
    final_input_df = final_input_df[feature_columns]

    return final_input_df


def predict_return():
    """
    Loads the saved model, prompts the user for input, preprocesses the data,
    makes a prediction, and displays the result.
    """
    print("\n--- Product Return Prediction Console Application ---")

    # Load the saved model and parameters
    try:
        model_components = joblib.load('lgbm_model_and_params(2).joblib')
        lgbm_tuned_model = model_components['model']
        feature_columns = model_components['feature_columns']
        optimal_threshold = model_components['optimal_threshold']  # This will now load the recall-optimized threshold
        print("Model and parameters loaded successfully.")
    except FileNotFoundError:
        print(
            "Error: 'lgbm_model_and_params.joblib' not found. Please ensure the model is saved and located correctly.")
        return
    except Exception as e:
        print(f"Error loading model components: {e}")
        return

    while True:
        user_input_data = {}
        print("\nPlease enter the following details for the product:")

        # Define features and their types/allowed values for validation
        input_features_meta = {
            'product_category': {'type': str,
                                 'options': ['Beauty', 'Fashion', 'Electronics', 'Sports', 'Automotive', 'Home',
                                             'Toys']},
            'product_price': {'type': float, 'range_desc': 'any positive number'},
            'quantity': {'type': int, 'range_desc': 'any positive integer'},
            'order_date': {'type': str, 'format_desc': 'YYYY-MM-DD'},
            'region': {'type': str,
                       'options': ['Europe', 'North America', 'Oceania', 'Africa', 'Asia', 'South America']},
            'payment_method': {'type': str, 'options': ['BankTransfer', 'CreditCard', 'Cash', 'PayPal']},
            'delivery_days': {'type': int, 'range_desc': 'any positive integer'},
            'customer_rating': {'type': float, 'range_desc': '2.0-5.0'},
            'discount_percent': {'type': int, 'range_desc': '0-20'},
            'revenue': {'type': float, 'range_desc': 'any positive number'}
        }

        for feature, meta in input_features_meta.items():
            while True:  # Outer loop for re-prompting on invalid input
                prompt_parts = []
                if 'range_desc' in meta: prompt_parts.append(meta['range_desc'])
                if 'options' in meta: prompt_parts.append(f"options: {', '.join(meta['options'])}")
                prompt_parts.append(f"format: {meta.get('format_desc', meta['type'].__name__)}")
                prompt_str = ", ".join(prompt_parts)
                user_value = input(f"Enter {feature} ({prompt_str}): ")

                try:
                    if meta['type'] == float:
                        parsed_value = float(user_value)
                    elif meta['type'] == int:
                        parsed_value = int(user_value)
                    else:
                        parsed_value = user_value

                    # Specific validations
                    if feature == 'order_date':
                        pd.to_datetime(parsed_value)  # Validate date format
                    elif 'options' in meta and parsed_value not in meta['options']:
                        raise ValueError(f"Invalid input. Must be one of: {', '.join(meta['options'])}")
                    elif feature == 'customer_rating' and not (2.0 <= parsed_value <= 5.0):
                        raise ValueError("Customer rating must be between 2.0 and 5.0.")
                    elif feature == 'discount_percent' and not (0 <= parsed_value <= 20):
                        raise ValueError("Discount percent must be between 0 and 20.")
                    elif feature in ['product_price', 'quantity', 'delivery_days', 'revenue'] and parsed_value <= 0:
                        raise ValueError(f"{feature} must be a positive number.")

                    user_input_data[feature] = parsed_value
                    break  # Exit inner loop if input is valid
                except ValueError as e:
                    print(f"Invalid input for {feature}: {e}. Please try again.")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}. Please try again.")

        try:
            processed_input = preprocess_input_data(user_input_data, feature_columns)

            # Make prediction
            prediction_proba = lgbm_tuned_model.predict_proba(processed_input)[:, 1]
            binary_prediction = (prediction_proba >= optimal_threshold).astype(int)

            print("\n--- Prediction Results ---")
            print(f"Probability of return: {prediction_proba[0]:.4f}")
            print(
                f"Predicted status: {'Returned' if binary_prediction[0] == 1 else 'Not Returned'} (based on recall-optimized threshold {optimal_threshold:.4f})")

        except Exception as e:
            print(f"An error occurred during prediction: {e}")

        another_prediction = input("\nDo you want to make another prediction? (yes/no): ").lower()
        if another_prediction != 'yes':
            break

    print("Exiting application. Goodbye!")


if __name__ == "__main__":
    predict_return()