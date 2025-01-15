from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
import pandas as pd
from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.predict_pipeline import CustomData, PredictPipeline  # Assuming this is where your prediction logic resides

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Pydantic model to parse form data for prediction
class PredictData(BaseModel):
    rainfall: float
    year: int
    irrigation_area: float
    crop_type: str
    soil_type: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": ""})

# Endpoint for single data prediction
@app.post("/predictdata", response_class=HTMLResponse)
async def predict_datapoint(request: Request, data: PredictData):
    custom_data = CustomData(
        year=data.year,
        rainfall=data.rainfall,
        irrigation_area=data.irrigation_area,
        crop_type=data.crop_type,
        soil_type=data.soil_type,
    )

    # Convert custom data into a DataFrame for prediction
    pred_df = custom_data.get_data_as_data_frame()
    # Initialize prediction pipeline
    predict_pipeline = PredictPipeline()
    prediction = predict_pipeline.predict(custom_data=pred_df)
    results = {"prediction": prediction}
    return JSONResponse(content=results)

# Endpoint to trigger the training pipeline
@app.post("/train", response_class=HTMLResponse)
async def train_model(request: Request):
    try:
        # Initialize and run the training pipeline
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        
        # Optionally, return a success message or any artifact
        return templates.TemplateResponse("home.html", {"request": request, "message": "Model training completed successfully!"})

    except Exception as e:
        # Handle errors in training
        return templates.TemplateResponse("home.html", {"request": request, "error": f"An error occurred during training: {str(e)}"})

