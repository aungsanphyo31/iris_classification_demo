import streamlit as st
import pickle
from sklearn.datasets import load_iris

# Load Iris data
iris = load_iris()

X = iris.data

# Load trained model
model = pickle.load(open("iris_model", "rb"))

# Sidebar
st.sidebar.title("Iris Flower Prediction")

sepal_length = st.sidebar.slider(
    "Sepal length (cm)",
    float(X[:, 0].min()),
    float(X[:, 0].max()),
    5.0
)

sepal_width = st.sidebar.slider(
    "Sepal width (cm)",
    float(X[:, 1].min()),
    float(X[:, 1].max()),
    3.0
)

petal_length = st.sidebar.slider(
    "Petal length (cm)",
    float(X[:, 2].min()),
    float(X[:, 2].max()),
    4.0
)

petal_width = st.sidebar.slider(
    "Petal width (cm)",
    float(X[:, 3].min()),
    float(X[:, 3].max()),
    1.0
)

# Main page
st.title("🌸 Iris Flower Classification")

# Classify button
if st.button("Classify"):

    data = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    result = model.predict(data)[0]

    flower_name = [
        "Iris Setosa",
        "Iris Versicolor",
        "Iris Virginica"
    ]

    st.write("### Prediction")
    st.success(f"The predicted flower is **{flower_name[result]}**")