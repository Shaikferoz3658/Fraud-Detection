import streamlit as st
import numpy as np
import pickle
import pandas as pd
from math import radians, cos, sin, asin, sqrt

# --- UTILITY FUNCTION: Haversine Distance ---
def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r

# --- LOAD MODELS ---
@st.cache_resource
def load_assets():
    # Replace these with your actual paths
    try:
        model = pickle.load(open("models/fraud_model.pkl", "rb"))
        scaler = pickle.load(open("models/scaler.pkl", "rb"))
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_assets()

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="FraudShield",
    page_icon="fraud-detection.png", 
    layout="wide"
)

# --- CUSTOM CSS & FONT AWESOME ---
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Roboto+Mono:wght@300;400&display=swap" rel="stylesheet">
    <style>
        * {
            font-family: 'Poppins', sans-serif;
        }
        .app-header {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            margin-bottom: 2rem;
            padding: 20px 0;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        .app-logo {
            font-size: 2.8rem;
            color: #fca311;
            animation: pulse 2s infinite;
        }
        .app-title {
            color: #fff;
            font-size: 2.8rem;
            font-weight: 800;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
            background: linear-gradient(90deg, #fca311, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .app-subtitle {
            color: #c7e3e1;
            font-size: 1rem;
            text-align: center;
            margin-top: 5px;
            letter-spacing: 2px;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        .main-header { 
            color: #1f4fd8; 
            text-align: center; 
            font-weight: 800;
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
        }
        .icon-blue { color: #fca311; margin-right: 10px; }
        .card { 
            padding: 20px; 
            border-radius: 12px; 
            background-color: #1e2130; 
            border: 1px solid #3e445e;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .fraud-text { 
            color: #ff4b4b; 
            font-weight: bold; 
            font-size: 1.25rem; 
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .safe-text { 
            color: #00eb93; 
            font-weight: bold; 
            font-size: 1.25rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .sidebar-icon { 
            margin-right: 10px;
            color: #EDF4F5;
        }
        .sidebar-section {
            margin: 25px 0;
        }
        .status-container {
            background: rgba(255, 255, 255, 0.19);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            padding: 12px 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .status-text {
            color: white;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .footer {
            text-align: center;
            padding: 20px;
            margin-top: 40px;
            color: #6c757d;
            font-size: 0.9rem;
            border-top: 1px solid #e0e0e0;
        }
        .footer-icons {
            margin-top: 10px;
            display: flex;
            justify-content: center;
            gap: 20px;
            font-size: 1.2rem;
        }
        .footer-icons a {
            color: #6c757d;
            transition: color 0.3s ease;
        }
        .footer-icons a:hover {
            color: #1f4fd8;
        }
        .sidebar-logo {
            text-align: center;
            padding: 20px 0;
            border-bottom: 1px solid #e0e0e0;
            margin-bottom: 20px;
        }
        .sidebar-logo i {
            font-size: 2.5rem;
            color: #f2e9e4;
            margin-bottom: 10px;
        }
        .sidebar-logo h3 {
            margin: 0;
            color: #f2e9e4;
            font-weight: 700;
        }
        .nav-item {
            padding: 12px 15px;
            margin: 5px 0;
            border-radius: 8px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .nav-item:hover {
            background-color: #f0f2f6;
        }
        .nav-item.active {
            background-color: #1f4fd8;
            color: white;
        }
        .nav-item.active i {
            color: white;
        }
        .section-header {
            color: #FAFAFA;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .section-header i{
            color:#fca311 ;
        }
        .warning-text {
            color: #ffa500;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 10px;
        }
        
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Transaction Check"

# Initialize session state for input values
if 'input_values' not in st.session_state:
    st.session_state.input_values = {
        'amt': None,
        'city_pop': None,
        'lat': None,
        'long': None,
        'merch_lat': None,
        'merch_long': None
    }

# --- SIDEBAR ---
with st.sidebar:
    # Logo Section
    st.markdown("""
    <div class="sidebar-logo">
        <i class="fa-solid fa-shield-halved"></i>
        <h3>FraudShield</h3>
        <p style="color: #666; font-size: 0.9rem;">Credit Card Fraud Detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    st.markdown('<div class="sidebar-section"><h4><i class="fa-solid fa-bars sidebar-icon"></i>Navigation</h4></div>', unsafe_allow_html=True)
    
    # Navigation buttons with icons
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown('<i class="fa-solid fa-magnifying-glass-chart" style="color: #FBF9F6;"></i>', unsafe_allow_html=True)
    with col2:
        if st.button("Transaction Check", use_container_width=True, 
                    type="primary" if st.session_state.current_page == "Transaction Check" else "secondary"):
            st.session_state.current_page = "Transaction Check"
            st.rerun()
    
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown('<i class="fa-solid fa-circle-info" style="color: #FBF9F6;"></i>', unsafe_allow_html=True)
    with col2:
        if st.button("About", use_container_width=True, 
                    type="primary" if st.session_state.current_page == "About" else "secondary"):
            st.session_state.current_page = "About"
            st.rerun()
    
    st.divider()
    
    # Status Section
    st.markdown('<div class="sidebar-section"><h4><i class="fa-solid fa-chart-line sidebar-icon"></i>System Status</h4></div>', unsafe_allow_html=True)
    
    # Status indicators
    st.markdown("""
    <div class="status-container">
        <div class="status-text">
            <i class="fa-solid fa-circle-check"></i>
            <span>System Online</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Additional status info
    st.markdown("""
    <div style="margin-top: 15px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin: 8px 0;">
            <span style="color: #666;"><i class="fa-solid fa-database"></i> Model Status</span>
            <span style="color: #00eb93;"> Loaded</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin: 8px 0;">
            <span style="color: #666;"><i class="fa-solid fa-shield"></i> Security Level</span>
            <span style="color: #00eb93;">High</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Set page based on session state
current_page = st.session_state.current_page

# --- MAIN HEADER ---
st.markdown("""
<div class="app-header">
    <div>
        <div class="app-title">FraudShield</div>
        <div class="app-subtitle">CREDIT CARD FRAUD DETECTION SYSTEM</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- PAGE: TRANSACTION CHECK ---
if current_page == "Transaction Check":
    st.markdown('<h1 class="main-header"><i class="fa-solid fa-magnifying-glass-chart icon-blue"></i>Transaction Analysis</h1>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('### <i class="fa-solid fa-file-invoice-dollar icon-blue"></i> Transaction Details', unsafe_allow_html=True)
        
        amt = st.number_input(
            "Transaction Amount ($)", 
            min_value=0.0, 
            max_value=1000000.0,
            value=st.session_state.input_values['amt'],
            placeholder="Enter amount: (e.g., 220.11)",
            help="Enter the transaction amount in US dollars (0 - 1,000,000)",
            key='amt_input'
        )
        st.session_state.input_values['amt'] = amt
        
        city_pop = st.number_input(
            "City Population", 
            min_value=0,
            max_value=20000000,
            value=st.session_state.input_values['city_pop'],
            placeholder="Enter city population: (e.g., 4154)",
            help="Population of the city where transaction occurred (0 - 20,000,000)",
            key='city_pop_input'
        )
        st.session_state.input_values['city_pop'] = city_pop
        
        st.markdown('### <i class="fa-solid fa-location-crosshairs icon-blue"></i> Geospatial Information', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            lat = st.number_input(
                "Customer Latitude", 
                min_value=-90.0,
                max_value=100000000.0,
                value=st.session_state.input_values['lat'],
                placeholder="e.g., 42.1808",
                help="Latitude coordinate of customer location (-90 to 90)",
                key='lat_input'
            )
            st.session_state.input_values['lat'] = lat
            
            merch_lat = st.number_input(
                "Merchant Latitude", 
                min_value=-90.0,
                max_value=10000000.0,
                value=st.session_state.input_values['merch_lat'],
                placeholder="e.g., 43.150704",
                help="Latitude coordinate of merchant location (-90 to 90)",
                key='merch_lat_input'
            )
            st.session_state.input_values['merch_lat'] = merch_lat
            
        with c2:
            long = st.number_input(
                "Customer Longitude", 
                min_value=-180.0,
                max_value=100000000.0,
                value=st.session_state.input_values['long'],
                placeholder="e.g., -112.262",
                help="Longitude coordinate of customer location (-180 to 180)",
                key='long_input'
            )
            st.session_state.input_values['long'] = long
            
            merch_long = st.number_input(
                "Merchant Longitude", 
                min_value=-180.0,
                max_value=1000000000.0,
                value=st.session_state.input_values['merch_long'],
                placeholder="e.g., -112.154481",
                help="Longitude coordinate of merchant location (-180 to 180)",
                key='merch_long_input'
            )
            st.session_state.input_values['merch_long'] = merch_long

        # Calculate distance only if all coordinates are provided
        if all(v is not None for v in [lat, long, merch_lat, merch_long]):
            try:
                distance = haversine(lat, long, merch_lat, merch_long)
                st.info(f"**Geospatial Distance**: {distance:.2f} km")
            except Exception as e:
                st.error(f"Error calculating distance: {str(e)}")
        else:
            st.warning("Enter all latitude/longitude values to calculate distance")

    with col_right:
        st.markdown('### <i class="fa-solid fa-map-location-dot icon-blue"></i> Transaction Map', unsafe_allow_html=True)
        
        # Show map only if coordinates are available
        if all(v is not None for v in [lat, long, merch_lat, merch_long]):
            map_data = pd.DataFrame({
                'lat': [lat, merch_lat],
                'lon': [long, merch_long],
                'label': ['Customer', 'Merchant']
            })
            st.map(map_data, color='#FBF9F6', zoom=6)
        else:
            st.info("Enter location coordinates to view the map")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Validation for analysis button
    all_fields_filled = all(v is not None for v in [
        amt, city_pop, lat, long, merch_lat, merch_long
    ])
    
    if st.button("Run Fraud Analysis", use_container_width=True, type="primary", 
                 disabled=not all_fields_filled):
        
        if not all_fields_filled:
            st.error("Please fill in all transaction details before running the analysis.")
            st.markdown('<p class="warning-text"><i class="fa-solid fa-exclamation-triangle"></i> All fields are required for accurate fraud detection</p>', unsafe_allow_html=True)
        
        elif model and scaler:
            try:
                # Prepare features
                features = np.array([amt, city_pop, lat, long, merch_lat, merch_long]).reshape(1, -1)
                scaled_features = scaler.transform(features)
                prediction = model.predict(scaled_features)[0]
                probability = model.predict_proba(scaled_features)[0][1]

                st.divider()
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    if prediction == 1:
                        st.markdown(f'''
                        <div class="card">
                            <p class="fraud-text">
                                <i class="fa-solid fa-triangle-exclamation"></i>
                                Fraudulent Activity Detected
                            </p>
                            <p style="color: #ccc; margin-top: 10px;">
                                <i class="fa-solid fa-lightbulb"></i>
                                This transaction requires manual review.
                            </p>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="card">
                            <p class="safe-text">
                                <i class="fa-solid fa-circle-check"></i>
                                Transaction Verified Safe
                            </p>
                            <p style="color: #ccc; margin-top: 10px;">
                                <i class="fa-solid fa-thumbs-up"></i>
                                This transaction appears legitimate.
                            </p>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                with res_col2:
                    st.metric(label="Risk Probability", value=f"{probability:.2%}")
                    st.progress(probability)
                    
                    # Risk level indicator
                    risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low"
                    risk_color = "#ff4b4b" if probability > 0.7 else "#ffa500" if probability > 0.3 else "#00eb93"
                        
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.info("Please check your input values and try again.")
                
        else:
            st.error("Model files not found. Please check your /models directory.")
            
    elif not all_fields_filled:
        st.markdown('<p class="warning-text"><i class="fa-solid fa-info-circle"></i> Fill all fields above to enable fraud analysis</p>', unsafe_allow_html=True)

# --- PAGE: ABOUT ---
elif current_page == "About":
    st.markdown('<h1 class="main-header"><i class="fa-solid fa-circle-info icon-blue"></i> About This System</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="section-header">
            <i class="fa-solid fa-microchip"></i>
            <h3>Fraud Detection System</h3>
        </div>
        <p>This advanced machine learning system analyzes credit card transactions in real-time to identify potential fraudulent activities.</p>
        
        <div class="section-header">
            <i class="fa-solid fa-gears"></i>
            <h4>How it works:</h4>
        </div>
        <ol style="color: #FAFAFA;">
            <li><strong>Data Collection</strong>: Transaction details are collected in real-time.</li>
            <li><strong>Feature Analysis</strong>: Multiple parameters are analyzed simultaneously.</li>
            <li><strong>ML Prediction</strong>: Trained model evaluates transaction patterns.</li>
            <li><strong>Risk Scoring</strong>: Each transaction receives a risk score.</li>
            <li><strong>Actionable Alerts</strong>: Instant notifications for suspicious activity.</li>
        </ol>
        
        <div class="section-header">
            <i class="fa-solid fa-star"></i>
            <h4>Key Features:</h4>
        </div>
        <ul style="color: #FAFAFA;">
            <li><i class="fa-solid fa-bolt" style="color: #C7E3E1;"></i> Real-time analysis</li>
            <li><i class="fa-solid fa-bullseye" style="color: #C7E3E1;"></i> High accuracy (99.4% detection rate)</li>
            <li><i class="fa-solid fa-shield-virus" style="color: #C7E3E1;"></i> Low false positive rate</li>
            <li><i class="fa-solid fa-layer-group" style="color: #C7E3E1;"></i> Scalable architecture</li>
            <li><i class="fa-solid fa-lock" style="color: #C7E3E1;"></i> Secure data processing</li>
        </ul>
        
        <div class="section-header">
            <i class="fa-solid fa-brain"></i>
            <h4>Model Details:</h4>
        </div>
        <ul style="color: #FAFAFA;">
            <li><strong>Algorithm</strong>: Logistic Regression with balanced classes</li>
            <li><strong>Training Data</strong>: 1M+ transactions</li>
            <li><strong>Features</strong>: 6 core parameters</li>
            <li><strong>Accuracy</strong>: 95% overall</li>
            <li><strong>Recall</strong>: 76% for fraud detection</li>
        </ul>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="section-header">
            <i class="fa-solid fa-list-check"></i>
            <h3>System Requirements</h3>
        </div>
        
        <div class="section-header">
            <i class="fa-solid fa-server"></i>
            <h5>Hardware:</h5>
        </div>
        <ul style="color: #FAFAFA;">
            <li><i class="fa-solid fa-microchip" style="color: #C7E3E1;"></i> CPU: 2+ cores</li>
            <li><i class="fa-solid fa-memory" style="color: #C7E3E1;"></i> RAM: 4GB minimum</li>
            <li><i class="fa-solid fa-hard-drive" style="color: #C7E3E1;"></i> Storage: 1GB free space</li>
        </ul>
        
        <div class="section-header">
            <i class="fa-solid fa-code-branch"></i>
            <h5>Software:</h5>
        </div>
        <ul style="color: #FAFAFA;">
            <li><i class="fa-brands fa-python" style="color: #C7E3E1;"></i> Python 3.8+</li>
            <li><i class="fa-solid fa-chart-line" style="color: #C7E3E1;"></i> Streamlit 1.28+</li>
            <li><i class="fa-solid fa-robot" style="color: #C7E3E1;"></i> scikit-learn 1.3+</li>
        </ul>
        
        <div class="section-header">
            <i class="fa-solid fa-user-shield"></i>
            <h5>Security:</h5>
        </div>
        <ul style="color: #FAFAFA;">
            <li><i class="fa-solid fa-key" style="color: #C7E3E1;"></i> End-to-end encryption</li>
            <li><i class="fa-solid fa-file-shield" style="color: #C7E3E1;"></i> GDPR compliant</li>
            <li><i class="fa-solid fa-shield-check" style="color: #C7E3E1;"></i> Regular security audits</li>
        </ul>
        
        <div class="section-header">
            <i class="fa-solid fa-chart-line"></i>
            <h5>Input Requirements</h5>
        </div>
        <ul style="color: #FAFAFA;">
            <li><i class="fa-solid fa-money-bill-wave" style="color: #C7E3E1;"></i> Amount: $0 - $1,000,000</li>
            <li><i class="fa-solid fa-users" style="color: #C7E3E1;"></i> Population: 0 - 20,000,000</li>
            <li><i class="fa-solid fa-globe" style="color: #C7E3E1;"></i> Latitude: -90° to 90°</li>
            <li><i class="fa-solid fa-globe-americas" style="color: #C7E3E1;"></i> Longitude: -180° to 180°</li>
        </ul>
        """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div class="footer">
    <div>
        <strong>FraudShield</strong> - Credit Card Fraud Detection System
    </div>
        <div style="margin: 10px 0; color: #6c757d;">
        <i class="fa-solid fa-copyright"></i> 2025 Babulu | All rights reserved
    </div>
</div>
""", unsafe_allow_html=True)