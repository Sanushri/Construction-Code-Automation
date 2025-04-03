import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st 
#import io


# Streamlit app
st.title("Construction Code Prediction")

# Upload Excel file
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Load the data from the uploaded Excel file
    prediction_data = pd.read_excel(uploaded_file,engine='openpyxl')
    url=r'https://github.com/Sanushri/Construction-Code-Automation/blob/main/Bm_construction_codesv2.xlsx'
    # Load the data from the Excel file "C:\Users\u1323736\OneDrive - MMC\Documents\PDCS -BM\Costruction Code Automation Project\Bm_construction_codesv2.xlsx"
    #file_path = r'C:\Users\u1323736\OneDrive - MMC\Documents\PDCS -BM\Costruction Code Automation Project\Bm_construction_codesv2.xlsx'
    # Replace with your file path
    data1 = pd.read_excel(url,engine='openpyxl')

    # Display the first few rows of the data
    print(data1.tail())

    data1['CONSTRUCTION DESCRIPTON']=data1['CONSTRUCTION DESCRIPTON'].astype(str)
    data1['Construction']=data1['Construction'].astype(str)



    # Assuming your DataFrame has columns 'CONSTRUCTION DESCRIPTION' and 'Construction Code'
    X = data1['CONSTRUCTION DESCRIPTON']
    y = data1['Construction']

    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)




    # Vectorize the text data
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)



 
    # Initialize the model
    model = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)

    # Train the model 
    model.fit(X_train_vectorized, y_train)

    # Make predictions
    y_pred = model.predict(X_test_vectorized)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')


    #prediction=r'C:\Users\u1323736\OneDrive - MMC\Documents\PDCS -BM\Costruction Code Automation Project\Bm_input.xlsx'
    #prediction_data=pd.read_excel(prediction)
    #prediction_data=prediction_data.fillna('Unknown')

    # New input data
    codes=[]
    # Vectorize the new input data
    new_descriptions_vectorized = vectorizer.transform(prediction_data['CONSTRUCTION DESCRIPTON'])

    # Make predictions
    predictions = model.predict(new_descriptions_vectorized)

    # Decode the predictions
    predicted_codes = label_encoder.inverse_transform(predictions)

    # Add predictions to the DataFrame
    prediction_data['Construction'] = predicted_codes

    prediction_data.head(2)

    # Save the updated DataFrame to a new Excel file
    output_file = "predictions.xlsx"
    #prediction_data.to_excel(output_file, index=False)

    # Save the updated DataFrame to a BytesIO object
    #output = io.BytesIO()
    pred=prediction_data.to_excel(output_file, index=False)
    #output.seek(0)  # Move the cursor to the beginning of the BytesIO object

    print(prediction_data)

    # Provide a download link
    st.download_button(
            label="Download Updated Excel File",
            data=pred,
            file_name="predictions.xlsx",
            mime="application/vnd.ms-excel"
            #mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    st.success("Predictions added successfully! You can download the updated file.")
else:
    st.error("The uploaded file must contain 'CONSTRUCTION DESCRIPTON' and 'Construction' columns.")















# Display the results
#for description, code in zip(prediction_data['CONSTRUCTION DESCRIPTON'], predicted_codes):
    #codes.append(code)
    #print(f'Description: {description} -> Construction Code: {code}')
    

#prediction_data.head(2)

#prediction_data.to_excel(prediction,index=False) 
