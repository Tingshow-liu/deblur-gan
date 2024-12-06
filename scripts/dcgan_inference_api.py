import torch
import torchvision.utils as vutils
from io import BytesIO
import base64
import os
import random
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "./output/netG_epoch_4.pth")
LATENT_VECTOR_SIZE = int(os.getenv("LATENT_VECTOR_SIZE", 100))


# Define the Generator model
class Generator(torch.nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(LATENT_VECTOR_SIZE, 512, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            torch.nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


# Load model on startup
@app.on_event("startup")
def load_model():
    global model, device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Generator(1).to(device)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()


class GenerateRequest(BaseModel):
    num_images: int
    seed: int = None  # Optional for reproducibility


@app.post("/generate/")
async def generate_images(request: GenerateRequest):
    # Set the seed for reproducibility
    if request.seed:
        random.seed(request.seed)
        torch.manual_seed(request.seed)

    # Generate random noise
    noise = torch.randn(request.num_images, LATENT_VECTOR_SIZE, 1, 1, device=device)

    # Generate images
    with torch.no_grad():
        generated_images = model(noise).cpu()

    # Convert images to base64 strings
    image_list = []
    for img_tensor in generated_images:
        buffer = BytesIO()
        vutils.save_image(img_tensor, buffer, format="PNG", normalize=True)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_list.append(image_base64)

    return {"images": image_list}


@app.get("/")
def health_check():
    return {"status": "Inference service is running."}
