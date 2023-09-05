from flask import Flask, render_template, request

app = Flask(__name__)

# Import necessary dependencies and define feature_extraction here
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load your mail_data as you did before
raw_mail_data = pd.read_csv('C:\\Users\\zuhay\\OneDrive\\Desktop\\mail_data.csv')
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

# Preprocess your mail_data, convert 'Category' to 0 and 1, and split into train/test
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

x = mail_data['Message']
y = mail_data['Category']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

# Define the feature_extraction (TfidfVectorizer)
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_train_features = feature_extraction.fit_transform(x_train)

# Convert y_train and y_test to integers
y_train = y_train.astype('int')
y_test = y_test.astype('int')

# Train the model
model = LogisticRegression()
model.fit(x_train_features, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check():
    input_mail = [request.form['text']]
    input_data_features = feature_extraction.transform(input_mail)
    prediction = model.predict(input_data_features)
    result = 'ham mail' if prediction[0] == 1 else 'spam mail'
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
