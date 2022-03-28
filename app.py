import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('/content/drive/My Drive/ATCSloans.pkl', 'rb')) 
# Feature Scaling
train = pd.read_excel('/content/drive/My Drive/FDP/train.xlsx')

# Extracting independent variable:
x = train.iloc[:, [1,2,3,4,5,6,7,8,9,10,11]].values
y = train.iloc[:, 12].values

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
x[:, 1] = labelencoder_x.fit_transform(x[:, 0])
x[:, 2] = labelencoder_x.fit_transform(x[:, 0])
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
x[:, 4] = labelencoder_x.fit_transform(x[:, 4])
x[:, 10] = labelencoder_x.fit_transform(x[:, 10])

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y[[0]] = labelencoder_y.fit_transform(y[[0]])

# Encoding the Independent Variable
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

#from sklearn.preprocessing import StandardScaler

#df = pd.read_csv('your file here')
# ss = StandardScaler()
# x_train = pd.DataFrame(ss.fit_transform(x_train),columns = x_train.columns)
# x_cv = pd.DataFrame(ss.fit_transform(x_cv),columns = x_cv.columns)

def predict_note_authentication(Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area):
  output= model.predict(sc.transform([[Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area]]))
  print("Loan_Status", output)
  if output==[1]:
    prediction="Loan will be given"
  else:
    prediction="Loan will not be given"
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:Grey;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:black;margin-top:10px;">Advanced Technology Consulting Service</p></center> 
   <center><p style="font-size:30px;color:black;margin-top:10px;">Transform Team</p></center> 
   <center><p style="font-size:25px;color:black;margin-top:10px;">Loan Approval Prediction</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Loan Approval Prediction")
    
    
    
    #Gender1 = st.select_slider('Select a Gender Male:1 Female:0',options=['1', '0'])
    Gender = st.number_input('Insert Gender Male:1 Female:0',0,1)
    Married = st.number_input('Insert Married Yes:1 No:0',0,1)
    Dependents = st.number_input('Insert Dependents',0,3)
    Education = st.number_input('Insert Education Graduate:1 Not Graduate:0',0,1)
    Self_Employed = st.number_input('Insert Self_Employed Yes:1 No:0',0,1)
    ApplicantIncome = st.number_input("Insert ApplicantIncome",1500,94106)
    CoapplicantIncome	 = st.number_input("Insert CoapplicantIncome",0,41667)
    LoanAmount = st.number_input('Insert LoanAmount',0,200000)
    Loan_Amount_Term = st.number_input('Inset Loan_Amount_Term', 12,480)
    Credit_History = st.number_input('Insert Credit_History',0,1)
    Property_Area = st.number_input('Insert Property_Area Urban:0 Rural:1 Semiurban:2',0,2)


    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area)
      st.success('Model has predicted {}'.format(result))
      
    if st.button("About"):
      st.subheader("Developed by Transfrom Team")
     
if __name__=='__main__':
  main()
