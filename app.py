from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')


load_dotenv()
API_KEY =os.environ['OPEN_API_KEY']

from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from pandasai import SmartDataframe

llm=OpenAI(api_token=API_KEY)


st.title("Data Analysis using Chat-GPT")


uploaded_csv = st.file_uploader("Upload a CSV fle", type=['csv'])

if uploaded_csv is not None:
    df=pd.read_csv(uploaded_csv)
    pandas_ai = SmartDataframe(df, config={"llm": llm})
    st.dataframe(df.head(3))

    prompt=st.text_area("Enter your prompt")
    if st.button("generate"):
        if prompt:
            with st.spinner("Generating answer....."):
                st.write(pandas_ai.chat(prompt))

            
        else:
            st.warning("Enter a prompt")
