from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd
from utils.logging import logger
from utils.exceptions import CustomException
from utils.helper import configs
from mlflow.sklearn import load_model
import mlflow
from schema import CustomerData

# Initialize FastAPI app
app = FastAPI()

# Mount static files (e.g., CSS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML template directory
templates = Jinja2Templates(directory="templates")

# Load your trained model (adjust the path as needed)
logged_model = configs['logged_model']

# Load model as a PyFuncModel.
model = joblib.load(logged_model)

# load preprocessor
preprocessor = joblib.load(configs['preprocessor_obj'])


# Route: Landing Page
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route: Form submission and prediction
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    credit_score: int = Form(...),
    country: str = Form(...),
    gender: str = Form(...),
    age: int = Form(...),
    tenure: int = Form(...),
    balance: float = Form(...),
    products_number: int = Form(...),
    credit_card: int = Form(...),
    active_member: int = Form(...),
    estimated_salary: float = Form(...)
):
    # Prepare input as DataFrame for prediction
    input_df = pd.DataFrame([{
        "credit_score": credit_score,
        "country": country,
        "gender": gender,
        "age": age,
        "tenure": tenure,
        "balance": balance,
        "products_number": products_number,
        "credit_card": credit_card,
        "active_member": active_member,
        "estimated_salary": estimated_salary
    }])
    

    pre_df=preprocessor.transform(input_df)
    final_df=pd.DataFrame(pre_df,columns=configs['all_columns'])
    print(final_df)

    # Make prediction
    pred = model.predict(final_df)[0]
    result = "Churn" if pred == 1 else "Stay"

    return templates.TemplateResponse("result.html", {"request": request, "result": result})

# Optional: run the app locally
if __name__ == "__main__":
    uvicorn.run("App:app", host="127.0.0.1", port=8000, reload=True)
