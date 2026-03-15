import streamlit as st
import joblib
import pandas as pd
import numpy as np
import datetime
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from db import init_db, save_prediction, save_dataset_records, get_prediction_history, get_uploaded_datasets

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

st.set_page_config(
    page_title="Product Return Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ────────────────────────────────────────────────
# BLUE THEME + MODERN METRIC CARDS
# ────────────────────────────────────────────────
st.markdown("""
    <style>
    :root {
        --primary-blue: #1d4ed8;
        --primary-dark: #1e40af;
        --primary-darker: #172554;
        --light-blue: #eff6ff;
        --blue-100: #dbeafe;
        --blue-200: #bfdbfe;
        --success-green: #10b981;
        --warning-red: #ef4444;
        --neutral-gray: #6b7280;
    }

    .main { 
        padding: 2rem 1rem; 
        background-color: var(--light-blue); 
    }

    h1, h2, h3 { color: var(--primary-blue); }

    .subtitle { color: var(--neutral-gray); }

    .stButton > button {
        background-color: var(--primary-blue) !important;
        color: white !important;
    }
    .stButton > button:hover {
        background-color: var(--primary-dark) !important;
    }

    hr { border-top: 2px solid var(--blue-200) !important; }

    /* Modern Metric Cards */
    div[data-testid="metric-container"] {
        background: white !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08) !important;
        padding: 1.4rem 1.6rem !important;
        position: relative !important;
        overflow: hidden !important;
        transition: all 0.25s ease !important;
        min-height: 110px !important;
    }

    /* result card styling for single prediction */
    .result-card {
        background: white;
        border-radius: 0.9rem;
        padding: 2rem 2.5rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        margin-top: 1.5rem;
    }
    .result-badge {
        font-size: 1.45rem;
        font-weight: 700;
        text-align: center;
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin: 1.3rem 0 2rem 0;
        border: 2px solid transparent;
    }
    .result-section {
        margin-bottom: 1.8rem;
    }

    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 12px 30px rgba(29, 78, 216, 0.15) !important;
    }

    div[data-testid="metric-container"]::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 5px;
        background: var(--primary-blue);
    }

    div[data-testid="metric-container"]:nth-child(1)::before { background: var(--primary-blue); }
    div[data-testid="metric-container"]:nth-child(2)::before { background: var(--warning-red); }
    div[data-testid="metric-container"]:nth-child(3)::before { background: var(--success-green); }

    div[data-testid="stMetricLabel"] > div {
        color: var(--neutral-gray) !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        margin-bottom: 0.5rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.4px !important;
    }

    div[data-testid="stMetricValue"] > div {
        color: #111827 !important;
        font-size: 2.4rem !important;
        font-weight: 700 !important;
        line-height: 1 !important;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        margin-top: 0.4rem !important;
    }

    div[data-testid="stMetricDelta"] svg {
        width: 16px !important;
        height: 16px !important;
    }

    [data-testid="stMetricDelta"][style*="color: rgb(0, 128, 0)"] { color: var(--success-green) !important; }
    [data-testid="stMetricDelta"][style*="color: rgb(255, 0, 0)"] { color: var(--warning-red) !important; }

    /* Slider fixes */
    section[data-testid="stSlider"] > div > div > div {
        background: var(--blue-200) !important;
    }
    [role="slider"][aria-valuenow] {
        background-color: var(--primary-blue) !important;
        border: 3px solid var(--primary-dark) !important;
    }

    /* Modal table styling (to match history) */
    .modal-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.94rem;
        color: #374151;
        background: white;
        border: 1px solid var(--blue-200);
        border-radius: 8px;
        overflow: hidden;
    }
    .modal-table thead {
        background: var(--blue-100);
    }
    .modal-table th {
        padding: 0.9rem 1.1rem;
        text-align: left;
        font-weight: 600;
        border-bottom: 1px solid var(--blue-200);
    }
    .modal-table th:nth-child(2), .modal-table th:nth-child(3) {
        text-align: right;
    }
    .modal-table th:nth-child(4) {
        text-align: center;
    }
    .modal-table td {
        padding: 0.9rem 1.1rem;
        border-bottom: 1px solid var(--blue-200);
    }
    .modal-table tr:last-child td {
        border-bottom: none;
    }
    .modal-table tr.returned {
        background: #fef2f2;
    }
    .modal-table tr.not-returned {
        background: #f0fdf4;
    }

    </style>
""", unsafe_allow_html=True)

init_db()

@st.cache_resource
def load_artifacts():
    try:
        artifact = joblib.load('lgbm_model_and_params(2).joblib')
        if isinstance(artifact, dict) and 'model' in artifact:
            model = artifact['model']
            optimal_threshold = artifact.get('optimal_threshold', 0.4917)
        else:
            model = artifact
            optimal_threshold = 0.4917
        return artifact, model, optimal_threshold
    except FileNotFoundError:
        st.error("Required model file not found.")
        st.stop()

artifact, model, optimal_threshold = load_artifacts()

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

DAY_NAMES = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
    4: "Friday", 5: "Saturday", 6: "Sunday"
}

def build_feature_vector(user_data, artifact):
    feature_cols = artifact.get('feature_columns', [])
    revenue = float(user_data.get('revenue', float(user_data['product_price']) * float(user_data['quantity'])))
    discount_percent = float(user_data.get('discount_percent', 0.0))
    order_value = float(user_data['product_price']) * float(user_data['quantity'])
    discount_amount = order_value * (discount_percent / 100.0)
    net_revenue = order_value - discount_amount
    order_date = user_data.get('order_date')
    if hasattr(order_date, 'month'):
        order_month = float(order_date.month)
        order_day = float(order_date.day)
        order_weekday = float(order_date.dayofweek)
    else:
        order_month = float(user_data.get('order_month', 1))
        order_day = float(user_data.get('order_day', 1))
        order_weekday = float(user_data.get('order_dayofweek', 0))

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
            vector.append(order_month)
        elif col == 'order_day':
            vector.append(order_day)
        elif col == 'order_weekday':
            vector.append(order_weekday)
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

def process_and_predict_batch(df, model, artifact):
    predictions_list = []
    first_error = None
    for idx, row in df.iterrows():
        try:
            order_date_raw = row.get('order_date', pd.Timestamp('2024-01-01'))
            order_date = pd.to_datetime(order_date_raw, errors='coerce', dayfirst=False)
            if pd.isna(order_date):
                raise ValueError("Invalid order_date format; expected YYYY-MM-DD.")
            order_month = int(order_date.month)
            order_dayofweek = int(order_date.dayofweek)

            user_data = {
                'product_category': str(row.get('product_category', 'Beauty')),
                'product_price': float(row.get('product_price', 0.0)),
                'quantity': int(row.get('quantity', 1)),
                'delivery_days': int(row.get('delivery_days', 1)),
                'customer_rating': float(row.get('customer_rating', 3.0)),
                'order_date': order_date,
                'region': str(row.get('region', 'North America')),
                'payment_method': str(row.get('payment_method', 'CreditCard')),
                'discount_percent': float(row.get('discount_percent', 0.0)),
                'revenue': float(row.get('revenue', float(row.get('product_price', 0.0)) * float(row.get('quantity', 1)))),
                'order_month': order_month,
                'order_dayofweek': order_dayofweek
            }
            processed = build_feature_vector(user_data, artifact)
            proba = model.predict_proba(processed)[0]
            prediction = int(proba[1] >= artifact.get('optimal_threshold', 0.5))
            save_prediction(user_data, prediction, proba, source="batch_upload")
            original_order_date = row.get('order_date', '')
            # Keep the original formatting for display while still using parsed date fields for features.
            if pd.isna(original_order_date):
                formatted_order_date = ''
            elif isinstance(original_order_date, (pd.Timestamp, datetime.datetime)):
                formatted_order_date = original_order_date.date().isoformat()
            elif isinstance(original_order_date, datetime.date):
                formatted_order_date = original_order_date.isoformat()
            else:
                formatted_order_date = str(original_order_date)

            predictions_list.append({
                'product_category': row.get('product_category', ''),
                'product_price': row.get('product_price', ''),
                'quantity': row.get('quantity', ''),
                'delivery_days': row.get('delivery_days', ''),
                'customer_rating': row.get('customer_rating', ''),
                'order_date': formatted_order_date,
                'prediction': 'Returned' if prediction == 1 else 'Not Returned',
                'prob_not_returned': f"{proba[0]:.4f}",
                'prob_returned': f"{proba[1]:.4f}"
            })
        except KeyError as e:
            missing_col = e.args[0] if e.args else 'unknown'
            first_error = f"Missing required column '{missing_col}'. Please add this column to your dataset."
            break
        except Exception as e:
            first_error = f"Could not process this data. Please verify your input values and required columns. Error: {str(e)}"
            break

    error_message = first_error
    result_df = pd.DataFrame(predictions_list) if predictions_list else None
    return result_df, error_message

# ────────────────────────────────────────────────
# HEADER & TABS
# ────────────────────────────────────────────────
st.markdown("<h1>Product Return Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Intelligent forecasting of product returns using machine learning</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Upload", "History"])

categories = [
    'Beauty', 'Electronics', 'Fashion', 'Home', 'Sports', 'Toys'
]

# ────────────────────────────────────────────────
# TAB 1: Single Prediction
# ────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3>Product Details</h3>", unsafe_allow_html=True)
        product_category = st.selectbox("Product Category", categories, key="cat_single")
        product_price = st.number_input("Product Price ($)", min_value=0.0, format="%.2f", key="price_single")
        quantity = st.number_input("Quantity (units)", min_value=1, step=1, key="qty_single")
        order_date = st.date_input("Order Date", key="order_date_single")
        region = st.selectbox("Region", options=["Europe", "North America", "Oceania", "Africa", "Asia", "South America"], index=1, key="region_single")

    with col2:
        st.markdown("<h3>Order Details</h3>", unsafe_allow_html=True)
        payment_method = st.selectbox("Payment Method", options=["BankTransfer", "CreditCard", "Cash", "PayPal"], index=1, key="payment_method_single")
        delivery_days = st.number_input("Delivery Days", min_value=0, step=1, key="deliv_single")
        customer_rating = st.slider("Customer Rating", 1.0, 5.0, 3.0, 0.1, key="rate_single")
        discount_percent = st.slider("Discount Percent", 0.0, 100.0, 0.0, 1.0, key="discount_single")
        revenue = product_price * quantity
        st.markdown(f"**Revenue (auto-calculated):** ${revenue:.2f}")

    modal_placeholder = st.empty()

    if st.button("Predict Return Risk", use_container_width=True, type="primary"):
        revenue = product_price * quantity
        order_date_dt = pd.to_datetime(order_date)
        user_data = {
            'product_category': product_category,
            'product_price': product_price,
            'quantity': quantity,
            'order_date': order_date_dt,
            'region': region,
            'payment_method': payment_method,
            'delivery_days': delivery_days,
            'customer_rating': customer_rating,
            'discount_percent': discount_percent,
            'revenue': revenue,
            'order_month': float(order_date_dt.month),
            'order_dayofweek': float(order_date_dt.dayofweek)
        }

        processed = build_feature_vector(user_data, artifact)
        proba = model.predict_proba(processed)[0]
        prob_return = proba[1]
        prob_keep = proba[0]
        prediction = int(prob_return >= optimal_threshold)

        save_prediction(user_data, prediction, proba, source="manual")

        # final label from the model's prediction (0/1) is used for the display
        is_return = prediction == 1
        decision_text = "Returned" if is_return else "Not Returned"
        decision_color = "#dc2626" if is_return else "#15803d"
        badge_bg = "#fee2e2" if is_return else "#f0fdfa"

        # risk levels are still based on return probability relative to threshold
        if prob_return >= 0.75:
            risk_level = "High Risk"
            risk_color = "#dc2626"
        elif prob_return >= optimal_threshold:
            risk_level = "Moderate Risk"
            risk_color = "#d97706"
        else:
            risk_level = "Low Risk"
            risk_color = "#15803d"

        row_class = "returned" if is_return else "not-returned"
        pred_display = "Returned" if is_return else "Not Returned"

        # build result card using Streamlit instead of custom modal
        card_html = f"""
        <div class='result-card'>
          <h2 style='margin:0 0 1.2rem; color:var(--primary-blue); text-align:center; font-weight:700;'>
            Prediction Result
          </h2>

          <div class='result-badge' style='background: {badge_bg}; color: {decision_color}; border-color: {decision_color}30;'>
            {decision_text}
          </div>

          <div class='result-section' style='display: flex; justify-content: space-between; gap: 1.5rem;'>
            <div style='text-align:center; flex:1;'>
              <div style='font-size:1rem; color:#4b5563; margin-bottom:0.4rem;'>Return Probability</div>
              <div style='font-size:2.8rem; font-weight:700; color:var(--primary-blue);'>{prob_return:.0%}</div>
            </div>
            <div style='text-align:center; flex:1;'>
              <div style='font-size:1rem; color:#4b5563; margin-bottom:0.4rem;'>Keep Probability</div>
              <div style='font-size:2.8rem; font-weight:700; color:#15803d;'>{prob_keep:.0%}</div>
            </div>
          </div>

          <div class='result-section' style='background:var(--blue-100); padding:1.2rem; border-radius:0.7rem; margin-bottom:1.8rem; font-size:0.98rem; line-height:1.5;'>
            <strong>Risk Level:</strong> <span style='color:{risk_color}; font-weight:700;'>{risk_level}</span><br>
            <strong>Decision threshold:</strong> ≥ {optimal_threshold:.4f} = predicted return
          </div>

          <h4 style='margin: 1.8rem 0 0.8rem; color:var(--primary-blue); font-weight:600;'>Prediction Details</h4>
          <table class='modal-table'>
            <thead>
              <tr>
                <th>Prediction</th>
                <th>Prob Keep</th>
                <th>Prob Return</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              <tr class='{row_class}'>
                <td>{pred_display}</td>
                <td style='text-align:right;'>{prob_keep:.4f}</td>
                <td style='text-align:right;'>{prob_return:.4f}</td>
                <td style='text-align:center;'>manual</td>
              </tr>
            </tbody>
          </table>
                                                                                                                                                                                                                                                                                
          <hr style='border-color:var(--blue-200); margin:2rem 0 1.5rem;'>

          <h4 style='margin:1.8rem 0 0.8rem; color:var(--primary-blue); font-weight:600;'>Input Details</h4>
          <table class='modal-table' style='margin-bottom:1.8rem;'>
            <tbody>
              <tr><td>Category</td><td><strong>{product_category}</strong></td></tr>
              <tr><td>Price</td><td><strong>${product_price:,.2f}</strong></td></tr>
              <tr><td>Quantity</td><td><strong>{quantity}</strong></td></tr>
              <tr><td>Delivery days</td><td><strong>{delivery_days}</strong></td></tr>
              <tr><td>Rating</td><td><strong>{customer_rating} / 5</strong></td></tr>
              <tr><td>Order Date</td><td><strong>{order_date.strftime('%Y-%m-%d')}</strong></td></tr>
            </tbody>
          </table>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

# ────────────────────────────────────────────────
# TAB 2: Batch Upload
# ────────────────────────────────────────────────
with tab2:
    st.markdown("<h2>Process Batch Data</h2>", unsafe_allow_html=True)
    st.write("Upload a CSV file with multiple product records for bulk prediction processing. All results will be stored in the database.")
    
    uploaded_file = st.file_uploader("Select CSV File", type="csv")
    
    if uploaded_file is not None:
        try:
            # store uploaded file in dedicated folder with timestamp
            import os, uuid
            save_dir = "uploaded_datasets"
            os.makedirs(save_dir, exist_ok=True)
            unique_name = f"upload_{pd.Timestamp.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.csv"
            save_path = os.path.join(save_dir, unique_name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            df = pd.read_csv(uploaded_file)

            if 'order_date' in df.columns:
                try:
                    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce').dt.date
                    parsed_dates = pd.to_datetime(df['order_date'], errors='coerce')
                    df['order_month'] = parsed_dates.dt.month
                    df['order_dayofweek'] = parsed_dates.dt.dayofweek
                except Exception as exc:
                    st.warning(f"Could not parse 'order_date' column: {exc}")

            st.markdown("<h3>Data Preview</h3>", unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)
            
            required_columns = ['product_category', 'product_price', 'quantity', 'delivery_days',
                              'customer_rating', 'order_date']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(
                    f"Dataset should contain needed columns: {', '.join(missing_columns)}. "
                    f"Please include these columns and re-upload."
                )
                st.info(f"Required columns: {', '.join(required_columns)}")
            else:
                st.success("CSV format is valid")
                
                if st.button("Process and Store in Database", use_container_width=True, type="primary"):
                    with st.spinner("Processing records..."):
                        save_dataset_records(df)
                        results_df, batch_error = process_and_predict_batch(df, model, artifact)
                        if batch_error:
                            st.error(batch_error)
                        elif results_df is not None:
                            st.success(f"Successfully processed {len(results_df)} records")
                            
                            st.markdown("<h3>Batch Results</h3>", unsafe_allow_html=True)
                            # Detailed records similar to history view
                            st.markdown("<h3 style='margin-top: 0.5rem;'>Detailed Records</h3>", unsafe_allow_html=True)
                            # show all columns with minimal formatting; include extra fields
                            st.dataframe(
                                results_df,
                                use_container_width=True,
                                column_config={
                                     "delivery_days": st.column_config.NumberColumn("Delivery", width="small"),
                                     "customer_rating": st.column_config.NumberColumn("Rating", format="%.1f", width="small"),
                                     "order_date": st.column_config.TextColumn("Order Date", width="small"),
                                     "prediction": st.column_config.TextColumn("Prediction", width="medium"),
                                     "prob_not_returned": st.column_config.NumberColumn("Prob Keep", format="%.4f"),
                                     "prob_returned": st.column_config.NumberColumn("Prob Return", format="%.4f"),
                                },
                                hide_index=True
                            )
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Records", len(results_df))
                            with col2:
                                returned_count = (results_df['prediction'] == 'Returned').sum()
                                st.metric("Returned", returned_count)
                            with col3:
                                not_returned_count = (results_df['prediction'] == 'Not Returned').sum()
                                st.metric("Not Returned", not_returned_count)
                            
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        else:
                            st.error("No records were processed.")
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# ────────────────────────────────────────────────
# TAB 3: History
# ────────────────────────────────────────────────
with tab3:
    st.markdown("<h2>Prediction History</h2>", unsafe_allow_html=True)
    st.write("View past predictions made through single entries or batch uploads.")
    
    limit_options = [50, 100, 250, 500, 1000]
    selected_limit = st.selectbox(
        "Number of records to display",
        options=limit_options,
        index=1,
        key="pred_limit_select"
    )
    
    history_df = get_prediction_history(selected_limit)
    
    if not history_df.empty:
        st.markdown("<h3 style='margin-bottom: 1rem;'>Summary</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Predictions",
                value=len(history_df),
                help="Total number of predictions in the selected range"
            )
        
        with col2:
            returned = (history_df['prediction'] == 1).sum()
            returned_pct = f"{returned / len(history_df):.1%}" if len(history_df) > 0 else "—"
            st.metric(
                label="Predicted Return",
                value=returned,
                delta=returned_pct,
                delta_color="inverse" if returned > 0 else "off"
            )
        
        with col3:
            not_returned = (history_df['prediction'] == 0).sum()
            not_returned_pct = f"{not_returned / len(history_df):.1%}" if len(history_df) > 0 else "—"
            st.metric(
                label="Predicted Keep",
                value=not_returned,
                delta=not_returned_pct,
                delta_color="normal"
            )
        
        st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
        
        st.markdown("<h3 style='margin-top: 0.5rem;'>Detailed Records</h3>", unsafe_allow_html=True)
        
        st.dataframe(
            history_df.drop(columns=["timestamp"], errors="ignore"),
            use_container_width=True,
            column_config={
                "prediction": st.column_config.TextColumn("Prediction", width="medium"),
                "prob_not_returned": st.column_config.NumberColumn("Prob Keep", format="%.4f"),
                "prob_returned": st.column_config.NumberColumn("Prob Return", format="%.4f"),
                "source": st.column_config.TextColumn("Source", width="small")
            },
            hide_index=True
        )
        
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="Download Full History as CSV",
            data=csv,
            file_name=f"prediction_history_{selected_limit}_records.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_history"
        )
    
    else:
        st.info("No prediction history available yet.\nMake some predictions (single or batch) to see records here.")
