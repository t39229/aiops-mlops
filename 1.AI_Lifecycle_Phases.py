### 1.1 Data Collection & Preprocessing ###

# ● Load data from a source
# ● Split into training and testing sets
# ● Standardize numerical values for consistency

import pandas as pd
# function from scikit-learn (sklearn), used to split data into training and testing sets.
from sklearn.model_selection import train_test_split
# Imports the StandardScaler class from scikit-learn, used for feature scaling.
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from prometheus_client import Counter, generate_latest
from flask import Flask, request, jsonify
import joblib
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV


# Load dataset from a CSV file
df = pd.read_csv(
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", header=None)


# Separate features (X) and target labels (y)
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]  # The last column (label)
# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)
# Standardize numerical features (mean=0, variance=1) for consistency
# Creates an instance of the StandardScaler.
# This scaler will standardize features by removing the mean and scaling to unit variance.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit & transform training data
X_test = scaler.transform(X_test)  # Transform testing data
# Note:
# ✔️ Ensures models receive clean, standardized data
# ✔️ Prevents features with larger numerical ranges from dominating others

#############################################################################################################################
### 1.2 Feature Engineering ###

# ● Select the most relevant features
# ● Improve model accuracy by eliminating irrelevant features

# Select top 5 features using the ANOVA F-value
selector = SelectKBest(score_func=f_classif, k=5)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

###############################################################################################################################
### 1.3 Model Training & Evaluation ###

# ● Train a model on labeled data
# ● Evaluate how well it performs

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_selected, y_train)  # Train model
# Predict on test data
y_pred = model.predict(X_test_selected)
# Measure model accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")

#############################################################################################################################
### 1.4 Hyperparameter Tuning ###
# Purpose
# ● Optimize model settings for better accuracy

# Define hyperparameter grid
param_grid = {"n_estimators": [50, 100, 150], "max_depth": [10, 20, None]}
# Perform grid search to find the best parameters
grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid, cv=3)
grid_search.fit(X_train_selected, y_train)
# Get the best model
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Why is this important?
# ✔️ Avoids underfitting/overfitting
# ✔️ Improves prediction accuracy

############################################################################################################################
### 2.2 Model Deployment ###
# These phases help automate, deploy, and monitor AI models.

### 2.1 Model Versioning ###

# ● Keep track of different trained models
# ● Allow easy rollback to previous models

# Log the trained model for versioning
mlflow.sklearn.log_model(best_model, "model")

# Why is this important?
# ✔️ Enables collaboration across teams
# ✔️ Ensures reproducibility

############################################################################################################################
# 2.2 Model Deployment
# Purpose
# ● Make the model available via an API
# Save the trained model
joblib.dump(best_model, "model.pkl")
# Create a Flask API
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Receive JSON input
    input_features = pd.DataFrame(data["features"])  # Convert to DataFrame


# Load model and predict
    loaded_model = joblib.load("model.pkl")
    predictions = loaded_model.predict(input_features)
    return jsonify({"predictions": predictions.tolist()})  # Return predictions


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")

# Why is this important?
# ✔️ Allows real-time predictions via API
# ✔️ Supports integration with applications

############################################################################################################################
### 2.3 Model Monitoring ###
# Purpose
# ● Track model performance in production

# Counter to track the number of predictions
prediction_counter = Counter("model_predictions", "Total Predictions Made")


@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest()


# Why is this important?
# ✔️ Ensures model reliability
# ✔️ Detects performance degradation
