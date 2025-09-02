import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import mlflow.sklearn

### 1.1 Data Collection & Preprocessing ###
# Purpose
# ● Load data from a source
# ● Split into training and testing sets
# ● Standardize numerical values for consistency


def train_model():
    # Load dataset from a CSV file
    df = pd.read_csv(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        header=None)

    # Separate features (X) and target labels (y)
    X = df.iloc[:, :-1]  # All columns except the last one
    y = df.iloc[:, -1]  # The last column (label)

    # Split data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Standardize numerical features (mean=0, variance=1) for consistency
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

###  Why is this important? ###
# ✔️ Ensures models receive clean, standardized data
# ✔️ Prevents features with larger numerical ranges from dominating others

    # Select top 5 features using the ANOVA F-value
    selector = SelectKBest(score_func=f_classif, k=5)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)


### Why is this important? ###
# ✔️ Reduces dimensionality, improving speed
# ✔️ Removes irrelevant or redundant features


### 1.3 Model Training & Evaluation ###
# Purpose
# ● Train a model on labeled data
# ● Evaluate how well it performs

    # Train a Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_selected, y_train)  # Train model

    # Define hyperparameter grid
    param_grid = {"n_estimators": [50, 100, 150], "max_depth": [10, 20, None]}
    # Perform grid search to find the best parameters
    grid_search = GridSearchCV(RandomForestClassifier(
        random_state=42), param_grid, cv=3)
    grid_search.fit(X_train_selected, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_selected)  # Predict on test data
    acc = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {acc:.2f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    
    ###  Why is this important? ###
# ✔️ Determines model effectiveness
# ✔️ Helps identify if further improvements are needed

### 2.2 Model Deployment ###
# Purpose
# ● Make the model available via an API

    # MLflow logging

    # Save model and preprocessing objects
    joblib.dump(best_model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(selector, "selector.pkl")
    
    
    try:
        mlflow.sklearn.log_model(best_model, "model")
        print("Model logged to MLflow")
    except Exception as e:
        print(f"MLflow logging failed: {e}")

    return best_model, scaler, selector, acc


if __name__ == "__main__":
    train_model()
    print("Training completed successfully!")
