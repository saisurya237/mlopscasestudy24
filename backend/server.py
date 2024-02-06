from io import StringIO
import pandas as pd
from transaction_classification import get_transaction
# from starlette.responses import Response
from fastapi.responses import JSONResponse

from fastapi import FastAPI, File, UploadFile, Response


app = FastAPI(
    title="Santandar Transactions Model",
    description="""Obtain transaction predictions for various entities that are part of the CSV input file""",
    version="0.1.0",
)


@app.post("/predict")
async def upload_csv(csv_file: UploadFile = File(...)):
    """Get classification of transactions from the csv file"""
    contents = await csv_file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))
    transaction_result = get_transaction(df)
    # Convert the DataFrame to a CSV-compatible format
    print("convert results df to csv")
    output = StringIO()
    transaction_result.to_csv(output, index=False)
    response = Response(content=output.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename=Results    "
    return response