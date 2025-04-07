# fastapi_app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import io
import json
import numpy as np
import uvicorn
from pyngrok import ngrok

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Define model architecture (same as in training)
def get_model(model_name, num_classes):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model

# Initialize the model
model_name = "resnet18"  # Default model
model = get_model(model_name, len(class_names))

# Load the trained model weights
try:
    model.load_state_dict(torch.load(f"models/{model_name}_best.pth", map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")

# Define image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/")
def read_root():
    return {"message": "Mint Leaf Classification API"}

@app.post("/classify/")
async def classify_image(file: UploadFile = File(...), model_name: str = "resnet18"):
    # Validate model name
    if model_name not in ["resnet18", "mobilenet_v2", "efficientnet_b0", "densenet121"]:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not supported")
    
    # Load model if different from current
    global model
    if model.__class__.__name__ != model_name:
        model = get_model(model_name, len(class_names))
        try:
            model.load_state_dict(torch.load(f"models/{model_name}_best.pth", map_location=torch.device('cpu')))
            model.eval()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    
    # Process the uploaded image
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top 3 predictions
            top3_prob, top3_catid = torch.topk(probabilities, 3)
            
            # Format results
            results = []
            for i in range(top3_prob.size(0)):
                results.append({
                    "class": class_names[top3_catid[i].item()],
                    "probability": round(top3_prob[i].item() * 100, 2)
                })
            
            is_mint = any(not cls["class"].startswith("non_mint") for cls in results 
                          if cls["probability"] > 50)
            
            return {
                "predictions": results,
                "is_mint": is_mint,
                "model_used": model_name
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Start Ngrok tunnel
ngrok.set_auth_token("NGROK_AUTH_TOKEN")
public_url = ngrok.connect(8000)  # Expose port 8000
print("Ngrok public URL:", public_url)



# Run Uvicorn server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # You would typically run this using uvicorn
    # e.g.: uvicorn fastapi_app:app --reload