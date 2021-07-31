# Import Uvicorn & the necessary modules from FastAPI
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException

# Import the PyCaret Regression module
import pycaret.classification as pycr

# Import other necessary packages
from dotenv import load_dotenv
import pandas as pd
import os

# Load the environment variables from the .env file into the application
load_dotenv() 

# Initialize the FastAPI application
app = FastAPI()

# Create a class to store the deployed model & use it for prediction
class Model:
    def __init__(self, modelname, bucketname):
        """
        Function to initalize the model
        modelname: Name of the model stored in the S3 bucket
        bucketname: Name of the S3 bucket
        """
        # Load the deployed model from Amazon S3
        self.model = pycr.load_model(modelname, platform = 'aws', authentication = { 'bucket' : bucketname })
    
    def predict(self, data):
        """
        Function to use the loaded model to make predictions on the data
        data: Pandas DataFrame to perform predictions
        """
        # After predicting, we return only the column containing the predictions (i.e. 'Label') after converting it to a list
        predictions = pycr.predict_model(self.model, data=data).Label.to_list()
        return predictions

# Load the model that you had deployed earlier on S3. Make sure enter your respective bucket name in place of 'mlopsdvc170100035'
model_et = Model("et_deployed", "mlopsdvc180070023")
model_rf = Model("rf_deployed", "mlopsdvc180070023")

col_list = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8',
'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
'Amount', 'Class']

# Create the POST endpoint with path '/predict'
@app.post("/et/predict")
# To understand how to handle file uploads in FastAPI, visit the documentation here
async def create_upload_file(file: UploadFile = File(...)):
    # Handle the file only if it is a CSV
    if file.filename.endswith(".csv"):
        
        # Create a temporary file with the same name as the uploaded CSV file so that the data can be loaded into a pandas Dataframe
        with open(file.filename, "wb")as f:
            f.write(file.file.read())
        data = pd.read_csv(file.filename)
        os.remove(file.filename)

        for col in col_list:
            if col not in data.columns:
                raise HTTPException(status_code=409, detail="Column not present or Invalid column name.")
                break
				
        # Return a JSON object containing the model predictions on the data
        return {
            "Labels": model_et.predict(data)
        }
    
    else:
        # Raise a HTTP 400 Exception, indicating Bad Request (you can learn more about HTTP response status codes here)
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted.")

@app.post("/rf/predict")
# To understand how to handle file uploads in FastAPI, visit the documentation here
async def create_upload_file(file: UploadFile = File(...)):
    # Handle the file only if it is a CSV
    if file.filename.endswith(".csv"):
        
        # Create a temporary file with the same name as the uploaded CSV file so that the data can be loaded into a pandas Dataframe
        with open(file.filename, "wb")as f:
            f.write(file.file.read())
        data = pd.read_csv(file.filename)
        os.remove(file.filename)
        
        for col in col_list:
            if col not in data.columns:
                raise HTTPException(status_code=409, detail="Column not present or Invalid column name.")
                break
                
        # Return a JSON object containing the model predictions on the data
        return {
            "Labels": model_rf.predict(data)
        }
    
    else:
        # Raise a HTTP 400 Exception, indicating Bad Request (you can learn more about HTTP response status codes here)
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted.")


# Check if the necessary environment variables for AWS access are available. If not, exit the program
if os.getenv("AWS_ACCESS_KEY_ID") == None or os.getenv("AWS_SECRET_ACCESS_KEY") == None:
    print("AWS Credentials missing. Please set required environment variables.")
    exit(1)