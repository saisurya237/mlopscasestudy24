import io
import pandas as pd
from transaction_classification import get_transaction
# from starlette.responses import Response
from fastapi.responses import JSONResponse

from fastapi import FastAPI, File, UploadFile


app = FastAPI(
    title="Santandar Transactions Model",
    description="""Obtain transaction predictions for various entities that are part of the CSV input file""",
    version="0.1.0",
)


@app.post("/predict")
async def upload_csv(csv_file: UploadFile = File(...)):
    """Get classification of transactions from the csv file"""
    df = pd.read_csv(csv_file.file)
    transaction_result = get_transaction(df)
    # Convert the DataFrame to a JSON-compatible format
    df_json = transaction_result.to_dict(orient="records")
    return JSONResponse(content=df_json)
