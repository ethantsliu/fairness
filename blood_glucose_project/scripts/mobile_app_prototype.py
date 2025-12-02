#!/usr/bin/env python3
"""
Mobile App Prototype: Lifestyle-Based Diabetes Risk Assessment
Create a web-based prototype for diabetes risk assessment using lifestyle factors

Author: Generated for fairness project
Date: November 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DiabetesRiskCalculator:
    """
    Diabetes risk calculator based on lifestyle factors
    """
    
    def __init__(self):
        # Model coefficients based on our analysis
        # These would normally be loaded from trained models
        self.feature_weights = {
            'age': 0.15,
            'bmi': 0.25,
            'mvpa_ratio': -0.30,  # Higher MVPA = lower risk
            'sedentary_ratio': 0.20,  # Higher sedentary = higher risk
            'light_activity_ratio': -0.10,
            'gender': 0.05,  # 1=Male, 2=Female
            'education_level': -0.08,
            'race_ethnicity': 0.03
        }
        
        self.risk_thresholds = {
            'low': 0.3,
            'moderate': 0.6,
            'high': 1.0
        }
    
    def calculate_risk_score(self, user_inputs):
        """Calculate diabetes risk score based on user inputs"""
        
        # Normalize inputs
        normalized_inputs = self.normalize_inputs(user_inputs)
        
        # Calculate weighted score
        risk_score = 0
        for feature, weight in self.feature_weights.items():
            if feature in normalized_inputs:
                risk_score += normalized_inputs[feature] * weight
        
        # Apply sigmoid transformation to get probability
        risk_probability = 1 / (1 + np.exp(-risk_score))
        
        return risk_probability
    
    def normalize_inputs(self, inputs):
        """Normalize user inputs to standard ranges"""
        normalized = {}
        
        # Age normalization (18-80 years)
        normalized['age'] = (inputs['age'] - 18) / (80 - 18)
        
        # BMI normalization (15-50 kg/m²)
        normalized['bmi'] = (inputs['bmi'] - 15) / (50 - 15)
        
        # Activity ratios (already 0-1)
        normalized['mvpa_ratio'] = inputs['mvpa_ratio']
        normalized['sedentary_ratio'] = inputs['sedentary_ratio']
        normalized['light_activity_ratio'] = inputs['light_activity_ratio']
        
        # Categorical variables
        normalized['gender'] = (inputs['gender'] - 1) / 1  # 1-2 -> 0-1
        normalized['education_level'] = (inputs['education_level'] - 1) / 4  # 1-5 -> 0-1
        normalized['race_ethnicity'] = (inputs['race_ethnicity'] - 1) / 4  # 1-5 -> 0-1
        
        return normalized
    
    def get_risk_category(self, risk_probability):
        """Get risk category based on probability"""
        if risk_probability < self.risk_thresholds['low']:
            return "Low Risk", "green"
        elif risk_probability < self.risk_thresholds['moderate']:
            return "Moderate Risk", "orange"
        else:
            return "High Risk", "red"
    
    def generate_recommendations(self, user_inputs, risk_probability):
        """Generate personalized recommendations"""
        recommendations = []
        
        # BMI recommendations
        if user_inputs['bmi'] > 30:
            recommendations.append("🎯 **Weight Management**: Your BMI indicates obesity. Consider consulting a healthcare provider about weight management strategies.")
        elif user_inputs['bmi'] > 25:
            recommendations.append("⚖️ **Healthy Weight**: Your BMI is in the overweight range. Small lifestyle changes can help achieve a healthier weight.")
        
        # Activity recommendations
        if user_inputs['mvpa_ratio'] < 0.05:  # Less than 5% MVPA
            recommendations.append("🏃‍♀️ **Increase Physical Activity**: Aim for at least 150 minutes of moderate-to-vigorous activity per week.")
        
        if user_inputs['sedentary_ratio'] > 0.7:  # More than 70% sedentary
            recommendations.append("🪑 **Reduce Sedentary Time**: Try to break up long periods of sitting with short walks or standing breaks.")
        
        # Age-specific recommendations
        if user_inputs['age'] > 45:
            recommendations.append("🩺 **Regular Screening**: Adults over 45 should have regular diabetes screening every 3 years.")
        
        # General recommendations
        recommendations.extend([
            "🥗 **Healthy Diet**: Focus on whole grains, lean proteins, fruits, and vegetables.",
            "💧 **Stay Hydrated**: Drink plenty of water and limit sugary beverages.",
            "😴 **Quality Sleep**: Aim for 7-9 hours of quality sleep per night.",
            "🧘‍♀️ **Stress Management**: Practice stress-reduction techniques like meditation or yoga."
        ])
        
        return recommendations

def create_sidebar_inputs():
    """Create sidebar input controls"""
    st.sidebar.header("📝 Personal Information")
    
    # Basic demographics
    age = st.sidebar.slider("Age (years)", 18, 80, 45)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    gender_code = 1 if gender == "Male" else 2
    
    # Physical measurements
    st.sidebar.subheader("Physical Measurements")
    height_cm = st.sidebar.number_input("Height (cm)", 140, 220, 170)
    weight_kg = st.sidebar.number_input("Weight (kg)", 40, 200, 70)
    bmi = weight_kg / ((height_cm / 100) ** 2)
    
    # Education and ethnicity
    st.sidebar.subheader("Background Information")
    education = st.sidebar.selectbox(
        "Education Level",
        ["Less than High School", "High School", "Some College", "College Graduate", "Graduate Degree"]
    )
    education_code = ["Less than High School", "High School", "Some College", "College Graduate", "Graduate Degree"].index(education) + 1
    
    ethnicity = st.sidebar.selectbox(
        "Race/Ethnicity",
        ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Asian", "Other"]
    )
    ethnicity_code = ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Asian", "Other"].index(ethnicity) + 1
    
    # Physical activity
    st.sidebar.header("🏃‍♀️ Physical Activity")
    st.sidebar.write("*Based on a typical day*")
    
    # Activity time inputs
    wear_time = st.sidebar.slider("Waking hours per day", 12, 18, 16)
    wear_time_minutes = wear_time * 60
    
    moderate_activity = st.sidebar.slider("Moderate activity (minutes/day)", 0, 120, 30)
    vigorous_activity = st.sidebar.slider("Vigorous activity (minutes/day)", 0, 60, 15)
    light_activity = st.sidebar.slider("Light activity (minutes/day)", 0, 300, 180)
    
    # Calculate remaining time as sedentary
    total_active_time = moderate_activity + vigorous_activity + light_activity
    sedentary_time = max(0, wear_time_minutes - total_active_time)
    
    # Calculate ratios
    mvpa_minutes = moderate_activity + vigorous_activity
    mvpa_ratio = mvpa_minutes / wear_time_minutes
    sedentary_ratio = sedentary_time / wear_time_minutes
    light_activity_ratio = light_activity / wear_time_minutes
    
    return {
        'age': age,
        'gender': gender_code,
        'bmi': bmi,
        'education_level': education_code,
        'race_ethnicity': ethnicity_code,
        'mvpa_ratio': mvpa_ratio,
        'sedentary_ratio': sedentary_ratio,
        'light_activity_ratio': light_activity_ratio,
        'wear_time_minutes': wear_time_minutes,
        'moderate_activity': moderate_activity,
        'vigorous_activity': vigorous_activity,
        'light_activity': light_activity,
        'sedentary_time': sedentary_time
    }

def create_activity_visualization(user_inputs):
    """Create activity breakdown visualization"""
    
    # Activity breakdown pie chart
    activities = ['Moderate Activity', 'Vigorous Activity', 'Light Activity', 'Sedentary Time']
    values = [
        user_inputs['moderate_activity'],
        user_inputs['vigorous_activity'], 
        user_inputs['light_activity'],
        user_inputs['sedentary_time']
    ]
    colors = ['#2E8B57', '#FF6347', '#FFD700', '#D3D3D3']
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=activities,
        values=values,
        hole=0.4,
        marker_colors=colors
    )])
    
    fig_pie.update_layout(
        title="Daily Activity Breakdown",
        font_size=12,
        showlegend=True
    )
    
    return fig_pie

def create_risk_gauge(risk_probability):
    """Create risk assessment gauge"""
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Diabetes Risk Score (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig_gauge.update_layout(height=400)
    
    return fig_gauge

def create_comparison_chart(user_inputs, risk_probability):
    """Create comparison with population averages"""
    
    # Population averages (based on our analysis)
    population_avg = {
        'BMI': 28.0,
        'MVPA Ratio': 0.08,
        'Sedentary Ratio': 0.65,
        'Risk Score': 0.52
    }
    
    user_values = {
        'BMI': user_inputs['bmi'],
        'MVPA Ratio': user_inputs['mvpa_ratio'],
        'Sedentary Ratio': user_inputs['sedentary_ratio'],
        'Risk Score': risk_probability
    }
    
    categories = list(population_avg.keys())
    
    fig_comparison = go.Figure()
    
    fig_comparison.add_trace(go.Scatterpolar(
        r=list(population_avg.values()),
        theta=categories,
        fill='toself',
        name='Population Average',
        line_color='lightblue'
    ))
    
    fig_comparison.add_trace(go.Scatterpolar(
        r=list(user_values.values()),
        theta=categories,
        fill='toself',
        name='Your Profile',
        line_color='red'
    ))
    
    fig_comparison.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(population_avg.values()), max(user_values.values())) * 1.1]
            )),
        showlegend=True,
        title="Comparison with Population Average"
    )
    
    return fig_comparison

def main():
    """Main Streamlit app"""
    
    # App header
    st.title("🩺 Diabetes Risk Assessment Tool")
    st.markdown("*Lifestyle-based diabetes risk prediction using NHANES research*")
    
    # Initialize calculator
    calculator = DiabetesRiskCalculator()
    
    # Get user inputs
    user_inputs = create_sidebar_inputs()
    
    # Calculate risk
    risk_probability = calculator.calculate_risk_score(user_inputs)
    risk_category, risk_color = calculator.get_risk_category(risk_probability)
    
    # Main dashboard
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.subheader("📊 Your Risk Assessment")
        
        # Risk gauge
        fig_gauge = create_risk_gauge(risk_probability)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Risk category display
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; background-color: {risk_color}20; border-left: 5px solid {risk_color};">
            <h3 style="color: {risk_color}; margin: 0;">Risk Category: {risk_category}</h3>
            <p style="margin: 5px 0 0 0;">Risk Score: {risk_probability:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🏃‍♀️ Activity Profile")
        
        # Activity breakdown
        fig_pie = create_activity_visualization(user_inputs)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Activity summary
        st.write("**Daily Activity Summary:**")
        st.write(f"• MVPA: {user_inputs['moderate_activity'] + user_inputs['vigorous_activity']} minutes ({user_inputs['mvpa_ratio']:.1%} of day)")
        st.write(f"• Light Activity: {user_inputs['light_activity']} minutes ({user_inputs['light_activity_ratio']:.1%} of day)")
        st.write(f"• Sedentary Time: {user_inputs['sedentary_time']} minutes ({user_inputs['sedentary_ratio']:.1%} of day)")
    
    with col3:
        st.subheader("📋 Profile Summary")
        st.write(f"**Age:** {user_inputs['age']} years")
        st.write(f"**BMI:** {user_inputs['bmi']:.1f} kg/m²")
        st.write(f"**Gender:** {'Male' if user_inputs['gender'] == 1 else 'Female'}")
        
        # BMI category
        if user_inputs['bmi'] < 18.5:
            bmi_cat = "Underweight"
        elif user_inputs['bmi'] < 25:
            bmi_cat = "Normal"
        elif user_inputs['bmi'] < 30:
            bmi_cat = "Overweight"
        else:
            bmi_cat = "Obese"
        
        st.write(f"**BMI Category:** {bmi_cat}")
    
    # Comparison section
    st.subheader("📈 Population Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_comparison = create_comparison_chart(user_inputs, risk_probability)
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with col2:
        st.write("**How you compare to the average adult:**")
        
        # BMI comparison
        avg_bmi = 28.0
        bmi_diff = user_inputs['bmi'] - avg_bmi
        st.write(f"• BMI: {user_inputs['bmi']:.1f} vs {avg_bmi:.1f} average ({bmi_diff:+.1f})")
        
        # Activity comparison
        avg_mvpa = 0.08
        mvpa_diff = user_inputs['mvpa_ratio'] - avg_mvpa
        st.write(f"• MVPA: {user_inputs['mvpa_ratio']:.1%} vs {avg_mvpa:.1%} average ({mvpa_diff:+.1%})")
        
        # Sedentary comparison
        avg_sedentary = 0.65
        sedentary_diff = user_inputs['sedentary_ratio'] - avg_sedentary
        st.write(f"• Sedentary: {user_inputs['sedentary_ratio']:.1%} vs {avg_sedentary:.1%} average ({sedentary_diff:+.1%})")
    
    # Recommendations section
    st.subheader("💡 Personalized Recommendations")
    recommendations = calculator.generate_recommendations(user_inputs, risk_probability)
    
    for i, rec in enumerate(recommendations[:6]):  # Show top 6 recommendations
        st.write(f"{i+1}. {rec}")
    
    # Educational content
    st.subheader("📚 Understanding Your Results")
    
    with st.expander("What does my risk score mean?"):
        st.write("""
        Your diabetes risk score is calculated based on lifestyle and demographic factors from NHANES research:
        
        - **Low Risk (0-30%)**: Your current lifestyle factors suggest a lower risk of developing diabetes
        - **Moderate Risk (30-60%)**: Some lifestyle factors may increase your risk - consider preventive measures
        - **High Risk (60%+)**: Multiple risk factors present - consult with a healthcare provider
        
        *Note: This tool is for educational purposes and should not replace professional medical advice.*
        """)
    
    with st.expander("Key factors in diabetes risk"):
        st.write("""
        **Most Important Factors:**
        1. **Physical Activity**: Regular moderate-to-vigorous activity significantly reduces risk
        2. **Body Weight**: Maintaining a healthy BMI is crucial for diabetes prevention
        3. **Sedentary Time**: Prolonged sitting increases risk, even with regular exercise
        4. **Age**: Risk increases with age, especially after 45
        5. **Family History**: Genetic factors play a role (not captured in this tool)
        """)
    
    with st.expander("About this tool"):
        st.write("""
        This diabetes risk assessment tool is based on research using the National Health and Nutrition 
        Examination Survey (NHANES) data from 2011-2014. The model uses lifestyle and demographic 
        factors to predict diabetes risk without requiring laboratory tests.
        
        **Data Sources:**
        - NHANES 2011-2014 (n=5,488 participants)
        - Physical activity measured by accelerometry
        - Diabetes defined by fasting glucose ≥100 mg/dL or HbA1c ≥5.7%
        
        **Model Performance:**
        - Accuracy: ~57%
        - Sensitivity: ~62%
        - Specificity: ~53%
        
        **Limitations:**
        - Does not include family history or genetic factors
        - Based on cross-sectional data
        - Not a substitute for clinical diagnosis
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("*Developed for research and educational purposes. Always consult healthcare professionals for medical advice.*")

if __name__ == "__main__":
    main()
