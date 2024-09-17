<<<<<<< HEAD
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import tree

# Load the Play Tennis dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Mild', 'Mild', 'Hot', 'Mild', 'Mild', 'Hot', 'Cool', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play Tennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Encode categorical features
df_encoded = pd.get_dummies(df, drop_first=True)

# Features and target variable
X = df_encoded.drop('Play Tennis_Yes', axis=1)
y = df_encoded['Play Tennis_Yes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title('Play Tennis Prediction App')

# User input features
st.sidebar.header('User Input Features')
def user_input_features():
    outlook = st.sidebar.selectbox('Outlook', options=['Sunny', 'Overcast', 'Rain'])
    temperature = st.sidebar.selectbox('Temperature', options=['Hot', 'Mild', 'Cool'])
    humidity = st.sidebar.selectbox('Humidity', options=['High', 'Normal'])
    wind = st.sidebar.selectbox('Wind', options=['Weak', 'Strong'])
    
    # Create a DataFrame for input features
    data = {
        'Outlook': outlook,
        'Temperature': temperature,
        'Humidity': humidity,
        'Wind': wind
    }
    return pd.DataFrame(data, index=[0])

input_data = user_input_features()
input_data_encoded = pd.get_dummies(input_data, drop_first=True)
input_data_encoded = input_data_encoded.reindex(columns=X.columns, fill_value=0)

# Prediction
prediction = model.predict(input_data_encoded)
prediction_proba = model.predict_proba(input_data_encoded)

# Display prediction results
st.subheader('Prediction Results')
st.write('Play Tennis:', 'Yes' if prediction[0] == 1 else 'No')
st.write('Prediction Probability:', prediction_proba)

# Visualize the decision tree
st.subheader('Decision Tree Visualization')
fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['No', 'Yes'], rounded=True, ax=ax)
st.pyplot(fig)
=======
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import tree

# Load the Play Tennis dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Mild', 'Mild', 'Hot', 'Mild', 'Mild', 'Hot', 'Cool', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play Tennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Encode categorical features
df_encoded = pd.get_dummies(df, drop_first=True)

# Features and target variable
X = df_encoded.drop('Play Tennis_Yes', axis=1)
y = df_encoded['Play Tennis_Yes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title('Play Tennis Prediction App')

# User input features
st.sidebar.header('User Input Features')
def user_input_features():
    outlook = st.sidebar.selectbox('Outlook', options=['Sunny', 'Overcast', 'Rain'])
    temperature = st.sidebar.selectbox('Temperature', options=['Hot', 'Mild', 'Cool'])
    humidity = st.sidebar.selectbox('Humidity', options=['High', 'Normal'])
    wind = st.sidebar.selectbox('Wind', options=['Weak', 'Strong'])
    
    # Create a DataFrame for input features
    data = {
        'Outlook': outlook,
        'Temperature': temperature,
        'Humidity': humidity,
        'Wind': wind
    }
    return pd.DataFrame(data, index=[0])

input_data = user_input_features()
input_data_encoded = pd.get_dummies(input_data, drop_first=True)
input_data_encoded = input_data_encoded.reindex(columns=X.columns, fill_value=0)

# Prediction
prediction = model.predict(input_data_encoded)
prediction_proba = model.predict_proba(input_data_encoded)

# Display prediction results
st.subheader('Prediction Results')
st.write('Play Tennis:', 'Yes' if prediction[0] == 1 else 'No')
st.write('Prediction Probability:', prediction_proba)

# Visualize the decision tree
st.subheader('Decision Tree Visualization')
fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['No', 'Yes'], rounded=True, ax=ax)
st.pyplot(fig)
>>>>>>> 49c7fde11ccd284ef92fcfd1b93ce1c721e992e2
