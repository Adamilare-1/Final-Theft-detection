# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 12:00:04 2025

@author: Adamilare
"""

# electricity_theft_detection_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PIL import Image
import time
import sys

# Load the model package with enhanced error handling
@st.cache_resource
def load_model():
    try:
        model_pkg = joblib.load('best_theft_model.pkl')
        
        # Validate the loaded package structure
        required_keys = ['model', 'scaler', 'feature_selector', 'features', 'metadata']
        if not all(key in model_pkg for key in required_keys):
            raise ValueError("Model package is missing required components")
            
        return model_pkg
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Function to get safe default values for inputs
def get_default_values(model_pkg):
    defaults = {}
    features = model_pkg['features']
    
    try:
        if hasattr(model_pkg['scaler'], 'mean_'):
            # Use scaler means if available
            defaults = {feat: float(model_pkg['scaler'].mean_[i]) 
                      for i, feat in enumerate(features)}
        else:
            # Fallback to zeros
            defaults = {feat: 0.0 for feat in features}
            
        return defaults
    except Exception as e:
        st.warning(f"Couldn't get default values: {str(e)}")
        return {feat: 0.0 for feat in features}

# Main function for the Streamlit app
def main():
    # Set page config
    st.set_page_config(
        page_title="Electricity Theft Detection",
        page_icon="⚡",
        layout="wide"
    )

    # Load model with error handling
    model_pkg = load_model()
    features = model_pkg['features']
    default_values = get_default_values(model_pkg)
    
    # Sidebar with model information
    with st.sidebar:
        st.title("⚡ Electricity Theft Detector")
        st.markdown("""
        **Deployed Model Details:**
        - Model Type: {}
        - ROC AUC: {:.4f}
        - Accuracy: {:.4f}
        - Model Size: {:.2f} KB
        - Features Used: {}
        """.format(
            model_pkg['metadata']['performance']['Model'],
            model_pkg['metadata']['performance']['Test ROC AUC'],
            model_pkg['metadata']['performance']['Accuracy'],
            model_pkg['metadata']['performance']['Model Size (KB)'],
            len(features)
        ))
        
        st.markdown("---")
        st.subheader("Quick Usage Guide")
        st.markdown("""
        1. **Manual Input**: Enter values for each feature
        2. **CSV Upload**: Upload a CSV file with customer data
        3. Click **Predict** to get results
        """)
        
        st.markdown("---")
        st.subheader("Feature Guide")
        st.markdown("""
        - Features may include positive and negative values
        - Use standardized units as per original data
        - Missing values will be filled with column medians
        """)

    # Main content area
    st.title("Electricity Theft Detection System")
    st.markdown("""
    This application helps utility companies identify potential electricity theft 
    based on consumption patterns and other features.
    """)

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Manual Input", "Batch Prediction (CSV)"])

    with tab1:
        st.subheader("Enter Customer Data Manually")
        
        # Create input fields for each feature with proper bounds
        input_data = {}
        cols = st.columns(3)
        
        # Split features into columns
        features_per_col = (len(features) + 2) // 3
        
        for i, feature in enumerate(features):
            with cols[i // features_per_col]:
                # Determine reasonable bounds based on feature name
                if "consumption" in feature.lower():
                    min_val = -1000.0
                    max_val = 10000.0
                    step = 1.0
                elif "diff" in feature.lower() or "delta" in feature.lower():
                    min_val = -500.0
                    max_val = 500.0
                    step = 0.1
                else:
                    min_val = -100.0
                    max_val = 100.0
                    step = 0.01
                
                input_data[feature] = st.number_input(
                    label=feature,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=default_values.get(feature, 0.0),
                    step=step,
                    format="%.4f",
                    help=f"Enter value for {feature}"
                )
        
        # Prediction button and results
        if st.button("Predict Theft Probability", key="manual_predict"):
            with st.spinner("Analyzing consumption patterns..."):
                try:
                    # Prepare input data
                    input_df = pd.DataFrame([input_data])
                    
                    # Validate all features are present
                    missing_features = set(features) - set(input_df.columns)
                    if missing_features:
                        raise ValueError(f"Missing features: {', '.join(missing_features)}")
                    
                    # Preprocess
                    X_selected = input_df[features]
                    X_scaled = model_pkg['scaler'].transform(X_selected)
                    
                    # Predict
                    if hasattr(model_pkg['model'], "predict_proba"):
                        proba = model_pkg['model'].predict_proba(X_scaled)[0][1]
                    else:
                        decision = model_pkg['model'].decision_function(X_scaled)[0]
                        proba = 1 / (1 + np.exp(-decision))  # Convert to probability
                    
                    # Display results
                    st.subheader("Prediction Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Theft Probability", f"{proba*100:.2f}%")
                        
                        if proba > 0.7:
                            st.error("High risk of electricity theft detected!")
                            st.markdown("**Recommended Action:** Schedule inspection")
                        elif proba > 0.4:
                            st.warning("Moderate risk of electricity theft")
                            st.markdown("**Recommended Action:** Monitor closely")
                        else:
                            st.success("Low risk of electricity theft")
                            st.markdown("**Recommended Action:** Normal monitoring")
                    
                    with col2:
                        # Visualize the probability
                        fig, ax = plt.subplots(figsize=(6, 1))
                        ax.barh(['Risk Level'], [proba], 
                               color='red' if proba > 0.5 else 'green')
                        ax.set_xlim(0, 1)
                        ax.set_xticks([0, 0.3, 0.7, 1])
                        ax.set_xticklabels(['0%', '30%', '70%', '100%'])
                        st.pyplot(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

    with tab2:
        st.subheader("Upload CSV for Batch Prediction")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file with customer data", 
            type="csv",
            help="CSV should contain all required features"
        )
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                batch_df = pd.read_csv(uploaded_file)
                
                # Show preview with scroll
                st.write("Data Preview (first 5 rows):")
                st.dataframe(batch_df.head())
                
                if st.button("Predict for Batch", key="batch_predict"):
                    with st.spinner("Processing batch predictions..."):
                        try:
                            # Check if all required features are present
                            missing_features = set(features) - set(batch_df.columns)
                            if missing_features:
                                raise ValueError(f"Missing required features: {', '.join(missing_features)}")
                            
                            # Preprocess
                            X_batch = batch_df[features]
                            X_batch_scaled = model_pkg['scaler'].transform(X_batch)
                            
                            # Predict
                            if hasattr(model_pkg['model'], "predict_proba"):
                                probabilities = model_pkg['model'].predict_proba(X_batch_scaled)[:, 1]
                            else:
                                decisions = model_pkg['model'].decision_function(X_batch_scaled)
                                probabilities = 1 / (1 + np.exp(-decisions))
                            
                            # Add predictions to dataframe
                            results_df = batch_df.copy()
                            results_df['Theft_Probability'] = probabilities
                            results_df['Risk_Level'] = pd.cut(
                                probabilities,
                                bins=[0, 0.4, 0.7, 1],
                                labels=['Low', 'Medium', 'High'],
                                include_lowest=True
                            )
                            
                            # Show results
                            st.subheader("Batch Prediction Results")
                            
                            # Add filtering
                            risk_filter = st.selectbox(
                                "Filter by risk level",
                                ["All", "High", "Medium", "Low"]
                            )
                            
                            if risk_filter != "All":
                                filtered_df = results_df[results_df['Risk_Level'] == risk_filter]
                            else:
                                filtered_df = results_df
                            
                            st.dataframe(filtered_df.sort_values('Theft_Probability', ascending=False))
                            
                            # Download button
                            csv = filtered_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label=f"Download {risk_filter} Risk Predictions",
                                data=csv,
                                file_name=f'theft_predictions_{risk_filter.lower()}_risk.csv',
                                mime='text/csv'
                            )
                            
                            # Summary statistics
                            st.subheader("Risk Distribution")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Total Customers", len(results_df))
                            high_risk = sum(results_df['Risk_Level'] == 'High')
                            col2.metric("High Risk Cases", 
                                       f"{high_risk} ({high_risk/len(results_df)*100:.1f}%)",
                                       delta=f"{high_risk} cases")
                            col3.metric("Avg Probability", 
                                       f"{results_df['Theft_Probability'].mean()*100:.1f}%",
                                       delta="Overall risk")
                            
                            # Risk distribution pie chart
                            risk_counts = results_df['Risk_Level'].value_counts()
                            fig, ax = plt.subplots()
                            ax.pie(risk_counts, 
                                  labels=risk_counts.index,
                                  autopct='%1.1f%%',
                                  colors=['green', 'orange', 'red'])
                            st.pyplot(fig)
                            
                        except Exception as e:
                            st.error(f"Batch prediction failed: {str(e)}")
                            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()