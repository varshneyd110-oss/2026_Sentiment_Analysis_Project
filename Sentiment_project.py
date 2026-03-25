import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np

def  mycleaning(doc):
    return re.sub("[^a-zA-Z ]","",doc).lower()


model=joblib.load("sentiment_model.pkl")

st.set_page_config(layout="wide")
st.markdown("""
    <div style="
        background: linear-gradient(90deg, #ffff00, #2E7D32);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    ">
        <h1 style="
            color: purple;
            font-size: 40px;
            margin: 0;
        ">
            Food Sentiment Analysis
        </h1>
    </div>
""", unsafe_allow_html=True)


st.sidebar.image("restaurant.jpg")

st.sidebar.title("About Project 🤖")
st.sidebar.write("AI-powered sentiment analysis for restaurant reviews")
st.sidebar.write("Decode customer emotions from restaurant reviews using AI.")


st.sidebar.title("Contact Us 📞")
st.sidebar.write("98765XXXXX")

st.sidebar.title("About Us 👥")
st.sidebar.write("Enter your review and let AI predict the sentiment")


st.write("\n")
st.write("## Enter Review")
sample=st.text_input("")
if st.button("Predict"):
    pred=model.predict([sample])

    prob=model.predict_proba([sample])
    if pred[0]==0:
        st.write("Negative 👎")
        st.write(f"Confidence Score : {prob[0][0]:.2f}")
    else:
        st.write("Positive 👍")
        st.write(f"Confidence Score : {prob[0][1]:.2f}")
        st.balloons()


st.write("#### Bulk Prediction")
file=st.file_uploader("select file",type=["csv","txt"])
if file:
    df=pd.read_csv(file,names=["Review"])
    placeholder=st.empty()
    placeholder.dataframe(df)
    if st.button("Predict",key="b2"):
        corpus=df.Review
        pred=model.predict(corpus)
        prob=np.max(model.predict_proba(corpus),axis=1)
        df["Sentiment"]=pred
        df["Confidence"]=prob
        df["Sentiment"]=df["Sentiment"].map({0:"Negative 👎",1:"Positive 👍"})
        placeholder.dataframe(df)
        st.balloons()





    

