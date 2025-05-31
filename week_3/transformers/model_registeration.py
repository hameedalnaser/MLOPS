import mlflow
import mlflow.sklearn
import pickle
import os
from pathlib import Path

import pandas as pd




@transformer
def transform(data, *args, **kwargs):
    # Extract model and DictVectorizer from input
    model = data['model']
    dv = data['dv']

    # Save DictVectorizer to file
    dv_path = "dv.pkl"
    with open(dv_path, "wb") as f_out:
        pickle.dump(dv, f_out)

    
    # Start MLflow run and log artifacts
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(dv_path)

        # After logging, find the MLmodel file and get its size
        run_id = mlflow.active_run().info.run_id
        run_dir = Path(f"mlruns/0/{run_id}/artifacts/model")
        mlmodel_file = run_dir / "MLmodel"

        if mlmodel_file.exists():
            size_bytes = os.path.getsize(mlmodel_file)
            print(f"MLModel file path: {mlmodel_file}")
            print(f"Model size (model_size_bytes): {size_bytes}")
        else:
            print("MLmodel file not found. Check logging path.")

    return "Model saved and size printed."
