import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("data/housing.csv")
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model configs
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(max_depth=5)
}
# Start tracking with MLflow
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_param("model", model_name)
        if model_name == "DecisionTree":
            mlflow.log_param("max_depth", 5)
        mlflow.log_metric("rmse", rmse)

        # Log model itself
        mlflow.sklearn.log_model(model, "model")
        print(f"{model_name} RMSE: {rmse:.4f}")