# Predicting Product Returns

This project uses a logistic regression model to predict whether an ordered item will be returned based on several features such as product category, price, delivery time, etc.

## Running the Streamlit App

1. Make sure you have the following files in the project root:
   - `best_lr_model_smote.joblib`
   - `scaler.joblib`
   - `label_encoder.joblib`

2. Install dependencies (example):
   ```bash
   pip install -r requirements.txt
   # or at least: pip install streamlit scikit-learn pandas joblib
   ```

3. Start the app:
   ```bash
   streamlit run app.py
   ```

4. A browser window should open with the interactive form. Fill in the values and press **Predict**.

## Command-line Interface (legacy)

You can still run the original CLI version by executing:
```bash
python main.py
```

This will prompt for inputs in the terminal.
