from flask import Flask,request,render_template
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['GET', 'POST'])
def predict():
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    return render_template('index.html',Accuracy=" Acuracy for iris dataset should be {} ".format(accuracy))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(port=5000,debug=True)