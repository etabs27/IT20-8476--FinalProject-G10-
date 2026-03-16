# Product Return Prediction System

A Streamlit-based web application for predicting product returns using a trained logistic regression model. The system supports both single predictions and batch processing with SQLite3 database for storing predictions and uploads.

## Features

✨ **Three Main Modes:**

1. **📝 Single Prediction** - Make predictions for individual products with an interactive form
2. **📊 Batch Upload** - Upload CSV files with multiple products to get predictions in bulk
3. **📈 History** - View and download prediction history and uploaded dataset records

💾 **Database Integration:**
- SQLite3 stores all predictions with timestamps and prediction probabilities
- Records uploaded dataset CSV files in the database
- Maintains complete audit trail of all predictions
- Download history data as CSV for analysis

## Installation

### Prerequisites
- Python 3.7+
- Pip package manager

### Setup

1. Navigate to the streamlit_app directory:
```bash
cd streamlit_app
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Ensure the following model files are in the `streamlit_app` directory:
   - `best_lr_model_smote.joblib`
   - `scaler.joblib`
   - `label_encoder.joblib`

## Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## Usage

### Single Prediction

1. Navigate to the **📝 Single Prediction** tab
2. Fill in the product details:
   - Product Category (dropdown)
   - Product Price
   - Quantity
   - Delivery Days
   - Customer Rating (1.0-5.0)
   - Order Month (1-12)
   - Order Day of Week (0=Monday, 6=Sunday)
3. Click **🔮 Predict**
4. View the prediction result with probability scores
5. Prediction is automatically saved to the database

### Batch Upload

1. Navigate to the **📊 Batch Upload** tab
2. Click on the file uploader to select a CSV file
3. The app will validate the file format
4. Review the data preview
5. Click **🚀 Process Batch & Store in Database**
6. View batch results with prediction statistics
7. Download results as CSV if needed

#### CSV Format Requirements

Your CSV file must contain these columns:
- `product_category` - String (Beauty, Fashion, Electronics, Home & Kitchen, Books, Toys, Automotive, Sports)
- `product_price` - Float
- `quantity` - Integer
- `delivery_days` - Integer
- `customer_rating` - Float (1.0-5.0)
- `order_month` - Integer (1-12)
- `order_dayofweek` - Integer (0-6, where 0=Monday)

**Example CSV:**
```csv
product_category,product_price,quantity,delivery_days,customer_rating,order_month,order_dayofweek
Beauty,29.99,2,5,4.5,3,2
Fashion,89.99,1,7,3.8,5,1
Electronics,199.99,1,3,4.2,2,4
```

A sample CSV file (`sample_data.csv`) is included in the `streamlit_app` directory for testing.

### History & Analytics

1. Navigate to the **📈 History** tab
2. View two sections:
   - **Prediction History**: All predictions made through the system (manual or batch)
   - **Uploaded Datasets**: All records from uploaded CSV files
3. Adjust the number of records to display (10-500)
4. View summary statistics
5. Download data as CSV for further analysis

## Database

The application uses SQLite3 (`streamlit_predictions.db`) to store:

### Predictions Table
- Timestamp of prediction
- All input features
- Prediction (0 = Not Returned, 1 = Returned)
- Probability scores for each class
- Source of prediction (manual or batch_upload)

### Uploaded Datasets Table
- Upload timestamp
- All input features from CSV
- Records the source dataset entries

## File Structure

```
streamlit_app/
├── app.py                           # Main Streamlit application
├── db.py                           # Database functions and initialization
├── best_lr_model_smote.joblib      # Trained ML model
├── scaler.joblib                   # Feature scaler
├── label_encoder.joblib            # Category encoder
├── streamlit_predictions.db        # SQLite database (auto-created)
├── requirements.txt                # Python dependencies
├── sample_data.csv                 # Sample CSV for testing
└── README.md                       # This file
```

## Model Information

- **Algorithm**: Logistic Regression (trained with SMOTE)
- **Features**: 7 input features
- **Optimal Threshold**: 0.4917
- **Output**: Binary classification (Returned/Not Returned)

## Troubleshooting

### File Not Found Error
Ensure the three joblib files are in the same directory as `app.py`

### CSV Upload Errors
- Check that all required columns are present
- Verify data types match requirements (numeric values shouldn't have text)
- Use the sample CSV as a template

### Database Issues
- The database is auto-generated on first run
- To reset: delete `streamlit_predictions.db` and restart the app
- The database location can be modified in `db.py`

## Development

To modify or extend the application:

1. **Add Features**: Edit `app.py` for UI changes
2. **Modify Database**: Edit `db.py` for schema changes
3. **Requirements**: Update `requirements.txt` if adding new packages

After making changes:
```bash
# Test syntax
python -m py_compile app.py db.py

# Restart the app
streamlit run app.py
```

## License

This project is part of the IT20-FinalProject by Group 10.

## Support

For issues or questions, refer to the project documentation or contact the development team.
