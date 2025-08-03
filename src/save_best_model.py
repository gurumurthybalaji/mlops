import mlflow
import joblib
import os

# Replace with the actual run_id of your best model
run_id = "bc96247551be4ef18a9f82170f43bebb"

# Download the model artifact
model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

# Save locally
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/best_model.pkl")
print("Best model saved to models/best_model.pkl")

