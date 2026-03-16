import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "streamlit_predictions.db"

def init_db():
    """Initialize SQLite database with tables for predictions and uploads."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Table for storing prediction records
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            product_category TEXT,
            product_price REAL,
            quantity INTEGER,
            delivery_days INTEGER,
            customer_rating REAL,
            order_month INTEGER,
            order_dayofweek INTEGER,
            prediction INTEGER,
            prob_not_returned REAL,
            prob_returned REAL,
            source TEXT
        )
    ''')
    
    # Table for storing uploaded dataset records
    c.execute('''
        CREATE TABLE IF NOT EXISTS uploaded_datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_timestamp TEXT,
            product_category TEXT,
            product_price REAL,
            quantity INTEGER,
            delivery_days INTEGER,
            customer_rating REAL,
            order_month INTEGER,
            order_dayofweek INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()

def save_prediction(user_data, prediction, proba, source="manual"):
    """Save prediction to database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO predictions 
        (timestamp, product_category, product_price, quantity, delivery_days, 
         customer_rating, order_month, order_dayofweek, prediction, prob_not_returned, prob_returned, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp,
        user_data['product_category'],
        user_data['product_price'],
        user_data['quantity'],
        user_data['delivery_days'],
        user_data['customer_rating'],
        user_data['order_month'],
        user_data['order_dayofweek'],
        int(prediction),
        float(proba[0]),
        float(proba[1]),
        source
    ))
    
    conn.commit()
    conn.close()

def save_dataset_records(df):
    """Save uploaded dataset records to database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for _, row in df.iterrows():
        c.execute('''
            INSERT INTO uploaded_datasets 
            (upload_timestamp, product_category, product_price, quantity, delivery_days, 
             customer_rating, order_month, order_dayofweek)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp,
            row['product_category'],
            row['product_price'],
            row['quantity'],
            row['delivery_days'],
            row['customer_rating'],
            row['order_month'],
            row['order_dayofweek']
        ))
    
    conn.commit()
    conn.close()

def get_prediction_history(limit=100):
    """Retrieve prediction history from database."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", 
        conn, 
        params=[limit]
    )
    conn.close()
    return df

def get_uploaded_datasets(limit=100):
    """Retrieve uploaded dataset records from database."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM uploaded_datasets ORDER BY upload_timestamp DESC LIMIT ?", 
        conn, 
        params=[limit]
    )
    conn.close()
    return df

def clear_predictions():
    """Clear all prediction records (admin function)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM predictions')
    conn.commit()
    conn.close()

def clear_uploads():
    """Clear all uploaded dataset records (admin function)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM uploaded_datasets')
    conn.commit()
    conn.close()
