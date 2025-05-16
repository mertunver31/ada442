import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Bank Marketing Prediction - Simple Demo",
    page_icon="💰",
    layout="wide"
)

def main():
    st.title("Bank Marketing Prediction App - Simple Test")
    
    st.write("""
    ## Test Deployment
    
    This is a simplified version of the Bank Marketing Prediction app to test Streamlit Cloud deployment.
    
    If this app loads correctly, then we can troubleshoot the full application.
    """)
    
    # Display a simple chart
    st.subheader("Sample Chart")
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['Term Deposit: Yes', 'Term Deposit: No', 'Overall Conversion']
    )
    st.line_chart(chart_data)
    
    # Display sample customer data
    st.subheader("Sample Customer Data")
    sample_data = {
        'age': [33, 42, 57, 28, 61],
        'job': ['admin', 'technician', 'retired', 'student', 'unemployed'],
        'marital': ['married', 'single', 'married', 'single', 'divorced'],
        'education': ['university', 'high.school', 'university', 'university', 'high.school'],
        'loan': ['no', 'yes', 'no', 'yes', 'no'],
        'contact': ['cellular', 'telephone', 'cellular', 'cellular', 'telephone'],
        'subscribed': ['no', 'no', 'yes', 'no', 'yes']
    }
    
    df = pd.DataFrame(sample_data)
    st.dataframe(df)

if __name__ == "__main__":
    main() 