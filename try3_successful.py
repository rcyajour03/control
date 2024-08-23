<<<<<<< HEAD
import streamlit as st
import pandas as pd
from interpret.glassbox import ClassificationTree
from interpret import show
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import re
import io
import sys

# Load the dataset and train the model
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree model
model = ClassificationTree()
model.fit(X_train, y_train)

# Streamlit UI
st.title('Decision Tree Classification App with Interpretability')

# User input features
st.sidebar.header('User Input Features')
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length', float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()))
    sepal_width = st.sidebar.slider('Sepal Width', float(X['sepal width (cm)'].min()), float(X['sepal width (cm)'].max()))
    petal_length = st.sidebar.slider('Petal Length', float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()))
    petal_width = st.sidebar.slider('Petal Width', float(X['petal width (cm)'].min()), float(X['petal width (cm)'].max()))

    # Create a DataFrame for input features
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    return pd.DataFrame(data, index=[0])

input_data = user_input_features()

# Make prediction
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Display prediction results
st.subheader('Prediction Results')
st.write('Predicted Class:', data.target_names[prediction[0]])
st.write('Prediction Probability:', prediction_proba)

# Explanation of predictions
st.subheader('Model Explanation')

# Capture the output of show() to extract the URL
class StreamToLogger(io.StringIO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.url = None

    def write(self, message):
        super().write(message)
        # Check if the message contains the URL pattern
        match = re.search(r'http://\S+', message)
        if match:
            self.url = match.group(0)

# Redirect stdout to capture output from show
logger = StreamToLogger()
sys.stdout = logger

# Show explainer in the browser
explainer = model.explain_local(X_train, y_train, name='Tree')
show(explainer)  # This opens the explanation in a browser tab

# Restore stdout
sys.stdout = sys.__stdout__

# Get the extracted URL
iframe_url = logger.url

# Embed the generated URL in an iframe if available
if iframe_url:
    st.components.v1.iframe(src=iframe_url, width=600, height=800)
else:
    st.write("No URL found for the decision tree explanation.")
=======
import streamlit as st
import pandas as pd
from interpret.glassbox import ClassificationTree
from interpret import show
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import re
import io
import sys

# Load the dataset and train the model
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree model
model = ClassificationTree()
model.fit(X_train, y_train)

# Streamlit UI
st.title('Decision Tree Classification App with Interpretability')

# User input features
st.sidebar.header('User Input Features')
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length', float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()))
    sepal_width = st.sidebar.slider('Sepal Width', float(X['sepal width (cm)'].min()), float(X['sepal width (cm)'].max()))
    petal_length = st.sidebar.slider('Petal Length', float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()))
    petal_width = st.sidebar.slider('Petal Width', float(X['petal width (cm)'].min()), float(X['petal width (cm)'].max()))

    # Create a DataFrame for input features
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    return pd.DataFrame(data, index=[0])

input_data = user_input_features()

# Make prediction
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Display prediction results
st.subheader('Prediction Results')
st.write('Predicted Class:', data.target_names[prediction[0]])
st.write('Prediction Probability:', prediction_proba)

# Explanation of predictions
st.subheader('Model Explanation')

# Capture the output of show() to extract the URL
class StreamToLogger(io.StringIO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.url = None

    def write(self, message):
        super().write(message)
        # Check if the message contains the URL pattern
        match = re.search(r'http://\S+', message)
        if match:
            self.url = match.group(0)

# Redirect stdout to capture output from show
logger = StreamToLogger()
sys.stdout = logger

# Show explainer in the browser
explainer = model.explain_local(X_train, y_train, name='Tree')
show(explainer)  # This opens the explanation in a browser tab

# Restore stdout
sys.stdout = sys.__stdout__

# Get the extracted URL
iframe_url = logger.url

# Embed the generated URL in an iframe if available
if iframe_url:
    st.components.v1.iframe(src=iframe_url, width=600, height=800)
else:
    st.write("No URL found for the decision tree explanation.")
>>>>>>> 49c7fde11ccd284ef92fcfd1b93ce1c721e992e2
