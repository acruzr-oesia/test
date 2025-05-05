# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {}
# META }

# CELL ********************

import mlflow

# Set given experiment as the active experiment. If an experiment with this name does not exist, a new experiment with this name is created.
# mlflow.set_experiment("experiment_prueba")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

!pip install imblearn

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import mlflow
import os

current_directory = os.getcwd()
tracking_path = os.path.join(current_directory, "mlrun")
#mlflow.set_tracking_uri("file://" + tracking_path)
#mlflow.set_registry_uri("file://" + tracking_path)
mlflow.set_tracking_uri("file:///tmp/mlruns")
mlflow.set_registry_uri("file:///tmp/mlruns")
print("traking ur configurada", mlflow.get_tracking_uri())

mlflow.set_experiment("experiment_prueba")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print("traking uri:", mlflow.get_tracking_uri)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
client = MlflowClient()

# Start your training job with `start_run()`
with mlflow.start_run() as run:

    # Entrena tu modelo
    lr = LogisticRegression()
    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])
    lr.fit(X, y)
    score = lr.score(X, y)
    signature = infer_signature(X, y)

    # Log metrics, params, and model
    mlflow.log_metric("score", score)
    mlflow.log_param("alpha", "alpha")
    mlflow.sklearn.log_model(lr, "model", signature=signature, registered_model_name="modelo_de_prueba")
    
    # Registra el modelo con un nombre diferente
    model_uri = "runs:/{}/model".format(run.info.run_id) # artefact
    print(run.info.run_id)

    run_id = run.info.run_id
    #model_uri = "runs:/1/model"
    mv = mlflow.register_model(model_uri, "modelo_prueba")

    #model_details = client.create_registered_model("modelo_prueba")
    client.create_model_version(name="modelo_prueba", source=model_uri, run_id=run_id)

    print(mv)
    print("Modelo registrado con nuevo nombre y run_id=%s" % run.info.run_id)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from mlflow.tracking import MlflowClient
client = MlflowClient()
models = client.search_registered_models()
for m in models:
    print(m.name)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


from mlflow.tracking import MlflowClient

# Nombre del modelo registrado
model_name = "modelo_prueba"

# Crear un cliente de MLflow
client = MlflowClient()

# Listar las versiones del modelo
model_versions = client.search_model_versions(f"name='{model_name}'")

# Mostrar las versiones del modelo ACTUALIZACIÓN
print("hola buenasss")
for version in model_versions:
    print(f"Versión: {version.version}, Estado: {version.current_stage}, URI: {version.source}")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
