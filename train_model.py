import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import mlflow.sklearn


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

    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=5)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    # Train initial model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_selected, y_train)

    # Hyperparameter tuning
    param_grid = {"n_estimators": [50, 100, 150], "max_depth": [10, 20, None]}
    grid_search = GridSearchCV(RandomForestClassifier(
        random_state=42), param_grid, cv=3)
    grid_search.fit(X_train_selected, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_selected)
    acc = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {acc:.2f}")
    print(f"Best Parameters: {grid_search.best_params_}")

    # Save model and preprocessing objects
    joblib.dump(best_model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(selector, "selector.pkl")

    # MLflow logging
    try:
        mlflow.sklearn.log_model(best_model, "model")
        print("Model logged to MLflow")
    except Exception as e:
        print(f"MLflow logging failed: {e}")

    return best_model, scaler, selector, acc


if __name__ == "__main__":
    train_model()
    print("Training completed successfully!")
