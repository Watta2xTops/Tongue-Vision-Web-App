from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from prediction import predict_disease
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify "http://localhost:3000"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    predicted_class = await predict_disease(file)
    print("Prediction result:", predicted_class)
    return JSONResponse(content={"prediction": predicted_class})
