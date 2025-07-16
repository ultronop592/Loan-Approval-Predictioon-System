# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 16:31:47 2025

@author: sraja
"""
import pandas as pd
import pickle



loaded_data = pickle.load(open('C:/Users/sraja/Downloads/Loan Prediction system/loan_data.sav','rb'))



input_data = ('LP001027','Male','Yes','2','Graduate','No','2500','1840','109','360','1','Urban' )

# Create a pandas DataFrame from the given data with the correct columns
# The columns should match the features used for training (excluding Loan_ID and Loan_Status)
input_data_df= pd.DataFrame([input_data], columns=['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'])

# Drop the 'Loan_ID' column as it was not used for training
input_data_df = input_data_df.drop(columns=['Loan_ID'], axis=1)

#converting categorical features to numerical for the given data
input_data_df.replace({'Married':{'No':0,'Yes':1}, 'Gender':{'Male':1,'Female':0}, 'Self_Employed':{'No': 0, 'Yes':1}, 'Property_Area':{'Rural':0,'Semiurban': 1, 'Urban': 2}, 'Education':{'Graduate': 1, 'Not Graduate':0}}, inplace = True)

#replacing 3+ dependecies to 4
input_data_df.replace(to_replace = '3+', value =4, inplace=True)

# Convert relevant columns to numeric, coercing errors
for col in ['Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
    input_data_df[col] = pd.to_numeric(input_data_df[col], errors='coerce')


# Make prediction on the preprocessed single instance
prediction = loaded_data.predict(input_data_df)

if (prediction[0] == 0):
  print('Loan Status: Not Approved')
else:
  print('Loan Status: Approved')