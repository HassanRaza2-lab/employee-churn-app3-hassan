import streamlit as st
import pandas as pd
import joblib
import json
import os

# --- Configuration ---
# Aapke naam se change karein
MY_NAME = "Hassan Raza" 

# --- 4. Load Schema and Model ---
try:
    # Load the trained pipeline
    pipeline = joblib.load('ml_pipeline.pkl')
    
    # Load the schema
    with open('schema.json', 'r') as f:
        schema = json.load(f)

    # Extract features from schema
    NUM_FEATURES = schema['numerical_features']
    CAT_FEATURES = schema['categorical_features']
    CAT_LEVELS = schema['categorical_levels']
    MODEL_LOADED = True
except FileNotFoundError:
    st.error("Error: 'ml_pipeline.pkl' or 'schema.json' files not found. Please make sure they are in the correct directory.")
    MODEL_LOADED = False
except Exception as e:
    st.error(f"An error occurred while loading files: {e}")
    MODEL_LOADED = False

# ----------------------------------------------------
# --- 5. Add your Name under Header as Sub-header ---
# ----------------------------------------------------
st.set_page_config(page_title="Employee Churn Predictor", layout="wide")
st.title("Employee Churn Prediction App 📊")
st.subheader(f"Created by {MY_NAME}")
st.markdown("---")


if MODEL_LOADED:
    st.success(f"ML Pipeline loaded successfully.")

    # Get imputed values for setting slider defaults
    try:
        num_imputer = pipeline.named_steps['preprocessor'].named_transformers_['num']['imputer']
    except:
        st.warning("Could not find Imputer statistics. Using fixed defaults.")
        # Fallback values if imputer structure is different
        num_imputer = type('DummyImputer', (object,), {'statistics_': [30, 2015, 2, 2]})()

    # Dictionary to store user inputs
    user_input = {}

    # ----------------------------------------------------
    # --- 6. Put Interactions in Streamlit for User ---
    # ----------------------------------------------------
    st.markdown("### Employee Input Data")
    col1, col2, col3 = st.columns(3)

    # Numerical Features Widgets
    with col1:
        # Age
        age_index = NUM_FEATURES.index('Age') if 'Age' in NUM_FEATURES else 0
        age_default_val = int(num_imputer.statistics_[age_index]) if age_index < len(num_imputer.statistics_) else 30
        user_input['Age'] = st.slider("Age", min_value=18, max_value=60, value=age_default_val, key='age')
        
        # JoiningYear
        year_index = NUM_FEATURES.index('JoiningYear') if 'JoiningYear' in NUM_FEATURES else 1
        year_default_val = int(num_imputer.statistics_[year_index]) if year_index < len(num_imputer.statistics_) else 2015
        user_input['JoiningYear'] = st.slider("Joining Year", min_value=2012, max_value=2018, value=year_default_val, key='joining')
        
    with col2:
        # PaymentTier
        tier_index = NUM_FEATURES.index('PaymentTier') if 'PaymentTier' in NUM_FEATURES else 2
        tier_default_val = int(num_imputer.statistics_[tier_index]) if tier_index < len(num_imputer.statistics_) else 2
        user_input['PaymentTier'] = st.selectbox("Payment Tier (1=Highest, 3=Lowest)", 
                                                options=[1, 2, 3], 
                                                index=min(max(tier_default_val - 1, 0), 2), 
                                                key='payment')
        
        # ExperienceInCurrentDomain
        exp_index = NUM_FEATURES.index('ExperienceInCurrentDomain') if 'ExperienceInCurrentDomain' in NUM_FEATURES else 3
        exp_default_val = int(num_imputer.statistics_[exp_index]) if exp_index < len(num_imputer.statistics_) else 2
        user_input['ExperienceInCurrentDomain'] = st.slider("Experience in Current Domain (Years)", 
                                                            min_value=0, max_value=7, value=exp_default_val, key='experience')
        
    # Categorical Features Widgets
    with col3:
        user_input['Education'] = st.selectbox("Education", options=CAT_LEVELS.get('Education', ['Bachelors', 'Masters', 'PHD']), key='education')
        user_input['City'] = st.selectbox("City", options=CAT_LEVELS.get('City', ['Bangalore', 'Pune', 'New Delhi']), key='city')
        user_input['Gender'] = st.selectbox("Gender", options=CAT_LEVELS.get('Gender', ['Male', 'Female']), key='gender')
        user_input['EverBenched'] = st.selectbox("Ever Benched", options=CAT_LEVELS.get('EverBenched', ['No', 'Yes']), key='benched')

    # ----------------------------------------------------
    # --- 7. Use Interaction Values for Prediction ---
    # ----------------------------------------------------
    
    # Construct DataFrame in the precise order used during training (NUM_FEATURES then CAT_FEATURES)
    input_order = NUM_FEATURES + CAT_FEATURES
    input_data = {col: [user_input.get(col)] for col in input_order}
    input_df = pd.DataFrame(input_data)

    st.markdown("---")
    
    # ----------------------------------------------------
    # --- 8. Show Model Prediction on Frontend on Submit Button ---
    # ----------------------------------------------------
    if st.button("Predict Employee Churn", type="primary"):
        with st.spinner('Predicting...'):
            try:
                prediction = pipeline.predict(input_df)[0]
                probabilities = pipeline.predict_proba(input_df)[0]
                prob_churn = probabilities[1] * 100
                
                st.markdown("#### Prediction Result:")
                
                if prediction == 1:
                    st.error(f"Prediction: **LIKELY TO LEAVE (CHURN)** 😔")
                else:
                    st.success(f"Prediction: **UNLIKELY TO LEAVE (STAY)** 😊")
                
                st.info(f"The model's confidence in **Churn (LeaveOrNot=1)** is **{prob_churn:.2f}%**.")

            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {e}")
                st.warning("Please check your input values. The ML pipeline might not be compatible with current inputs.")