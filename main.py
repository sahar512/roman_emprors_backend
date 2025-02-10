from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import io
from fastapi.middleware.cors import CORSMiddleware

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")  # Prints "cuda" if GPU is being used

# **Define class labels (Must match training)**
class_names = [
    "Vespasian", "Hadrian", "Trajan", "Antoninus", "Nerva", "Pertinax",
    "Alexander", "Vitellius", "Augustus", "Caligula", "Caracalla", "Claudius",
    "Commodus", "Didius", "Domitian", "Elagabalus", "Galba", "Geta",
    "Lucius", "Macrinus", "Marcus", "Nero", "Otho", "Septimius",
    "Tiberius", "Titus", "Maximinus Thrax", "Pupienus", "Balbinus",
    "Gordian III", "Philip the Arab", "Decius", "Trebonianus Gallus",
    "Aemilian", "Valerian", "Gallienus", "Claudius Gothicus", "Quintillus",
    "Aurelian", "Tacitus", "Florian", "Probus", "Carus", "Numerian",
    "Carinus", "Diocletian", "Maximian", "Constantius I", "Galerius",
    "Severus II", "Gordian II AND I"
]

# **Define the EfficientNet-B3 model architecture (MUST MATCH TRAINING)**
def load_model():
    model = models.efficientnet_b3(weights=None)  # No pre-trained weights
    model.classifier[1] = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.classifier[1].in_features, len(class_names))  # 53 classes
    )

    # **Load trained weights**
    model.load_state_dict(torch.load("roman_emperors_FINAL.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

# **Load trained model**
model = load_model()

# **Define image preprocessing (Must match training)**
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure correct input size
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # Match training normalization
])

# **Create FastAPI app**
app = FastAPI()

# **Enable CORS for React Frontend**
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow frontend access
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        print(f"📂 Received file: {file.filename}")  # Debugging

        # Read and preprocess image
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension
        image = image.to(device)  # Move image tensor to GPU (if available)

        # **Make prediction**
        with torch.no_grad():
            output = model(image)  # Model runs on GPU
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_class = torch.max(probabilities, 0)

        # **Return result**
        result = {
            "emperor": class_names[predicted_class.item()],
            "confidence": round(confidence.item() * 100, 2)  # Convert to percentage
        }
        print(f"✅ Prediction: {result}")  # Debugging
        return result

    except Exception as e:
        print(f"❌ Error: {str(e)}")  # Debugging
        return {"error": str(e)}

# **Run the API (for local testing)**
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)  # Change host to 127.0.0.1
