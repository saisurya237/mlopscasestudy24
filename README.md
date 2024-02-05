# mlopscasestudy24 - Santander Customer Transaction Classification.

## Pre-requisites to install for Local run.
1. Poetry - Python package manager [Poetry](https://python-poetry.org/docs/#installation)
2. Python - 3.9
3. MLflow server - [MLflow](https://mlflow.org/docs/latest/tracking/tutorials/local-database.html)
4. Docker Compose - [DockerCompose](https://docs.docker.com/compose/install/)

## Steps to run the application locally.
1. Set the `mlflow.set_tracking_uri(<local mlflow server>)` in the "training.py" file in "model_training" folder
2. in the root folder, run `poetry install` where the "pyproject.toml" is present.
3. Activate the virtual env - `pathtothevirtualenv\Scripts\activate.ps1` for powershell.
4. Train the model - `python ./training.py`
5. in the frontend/ui.py file modify the backend variable to the localhost backend url. example - `http://localhost:8000` 
6. Run the following commands to deploy the backend and the frontend locally `docker-compose build` & `docker-compose up`

Notes - 
1. The containers will spin up, the backend service should be available at `http://localhost:8000/docs` for backend testing.
2. The frontent service should be available at `http://localhost:8501`
