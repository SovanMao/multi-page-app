import streamlit as st
import pandas as pd
from sklearn import datasets

def app():
    st.title('Iris Dataset Summary')
    iris = datasets.load_iris()

    X = pd.DataFrame(iris.data, columns = iris.feature_names)
    Y = pd.Series(iris.target, name = 'class')

    df = pd.concat([X,Y], axis=1)
    df['class'] = df['class'].map({0:"setosa", 1:"versicolor", 2:"virginica"})

    summary = df.describe(include='all')
    st.write(summary)

    # Randomly Display Sample Data
    st.write("## Sample Data")
    st.write("Randomly Display Sample Data")
    sample_size = st.slider("Select sample size", min_value=1, max_value=len(df), value=min(len(df), 10))
    st.write(df.sample(sample_size))  

