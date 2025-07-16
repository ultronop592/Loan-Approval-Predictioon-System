# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 17:58:05 2025

@author: sraja
"""

import pandas as pd 
import pickle
import streamlit as st 

loaded_data = pickle.load(open('loan_data.sav','rb'))

def loan_Approval(input_data):
    # Create a pandas DataFrame from the given data with the correct columns
    input_data_df = pd.DataFrame([input_data], columns=[
       'Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area'
   ])
    
    # Drop the 'Loan_ID' column as it was not used for training
    input_data_df = input_data_df.drop(columns=['Loan_ID'], axis=1)
    
    # Handle empty strings before conversion
    # Replace empty strings with appropriate defaults
    input_data_df = input_data_df.replace('', pd.NA)
    
    # Fill missing values with defaults
    defaults = {
        'Gender': 'Male',
        'Married': 'No', 
        'Dependents': '0',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 0,
        'CoapplicantIncome': 0,
        'LoanAmount': 0,
        'Loan_Amount_Term': 360,  # Common loan term
        'Credit_History': 0,
        'Property_Area': 'Urban'
    }
    
    input_data_df = input_data_df.fillna(defaults)
    
    # Convert categorical features to numerical
    input_data_df.replace({
        'Married': {'No': 0, 'Yes': 1},
        'Gender': {'Male': 1, 'Female': 0},
        'Self_Employed': {'No': 0, 'Yes': 1},
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
        'Education': {'Graduate': 1, 'Not Graduate': 0}
    }, inplace=True)
    
    # Replace '3+' with 4 in Dependents
    input_data_df.replace(to_replace='3+', value=4, inplace=True)
    
    # Convert relevant columns to numeric
    numeric_columns = ['Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    for col in numeric_columns:
        input_data_df[col] = pd.to_numeric(input_data_df[col], errors='coerce')
        # Fill any remaining NaN values with 0 (except for Credit_History and Loan_Amount_Term)
        if col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Dependents']:
            input_data_df[col] = input_data_df[col].fillna(0)
        elif col == 'Loan_Amount_Term':
            input_data_df[col] = input_data_df[col].fillna(360)
        elif col == 'Credit_History':
            input_data_df[col] = input_data_df[col].fillna(0)
    
    try:
        # Make prediction
        prediction = loaded_data.predict(input_data_df)
        if prediction[0] == 0:
            return 'Loan Status: Not Approved'
        else:
            return 'Loan Status: Approved'
    except Exception as e:
        return f'Error in prediction: {str(e)}'

def main():
    st.title('Loan Approval Prediction System')
    
    # Input fields with better user experience
    Loan_ID = st.text_input('Loan ID Number')
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Married = st.selectbox('Marital Status', ['No', 'Yes'])
    Dependents = st.selectbox('Number of Dependents', ['0', '1', '2', '3+'])
    Education = st.selectbox('Education Level', ['Graduate', 'Not Graduate'])
    Self_Employed = st.selectbox('Self Employed', ['No', 'Yes'])
    ApplicantIncome = st.text_input('Applicant Income', value='0')
    CoapplicantIncome = st.text_input('Co-applicant Income', value='0')
    LoanAmount = st.text_input('Loan Amount', value='0')
    Loan_Amount_Term = st.text_input('Loan Amount Term (months)', value='360')
    Credit_History = st.selectbox('Credit History', ['0', '1'])
    Property_Area = st.selectbox('Property Area', ['Rural', 'Semiurban', 'Urban'])
    
    approval = ''
    
    if st.button('Loan Approval Result'):
        approval = loan_Approval([
            Loan_ID, Gender, Married, Dependents, Education, Self_Employed,
            ApplicantIncome, CoapplicantIncome, LoanAmount,
            Loan_Amount_Term, Credit_History, Property_Area
        ])
        st.success(approval)

if __name__ == '__main__':
    main()
