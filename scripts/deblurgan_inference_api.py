from PIL import Image, ImageOps
from fastapi import FastAPI, HTTPException, UploadFile, File
import torch
import torchvision.transforms as transforms
import os

# Define the FastAPI app
app = FastAPI()

# Load environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "./output/generator_epoch_300.pth")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
INPUT_NC = int(os.getenv("INPUT_NC", 3))
OUTPUT_NC = int(os.getenv("OUTPUT_NC", 3))
NGF = int(os.getenv("NGF", 64))
N_BLOCKS = int(os.getenv("N_BLOCKS", 6))


# Define the ResNet Block
class ResnetBlock(torch.nn.Module):
    def __init__(
        self,
        dim,
        padding_type="reflect",
        norm_layer=torch.nn.BatchNorm2d,
        use_dropout=False,
        use_bias=True,
    ):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        if padding_type == "reflect":
            conv_block += [torch.nn.ReflectionPad2d(1)]
        else:
            raise NotImplementedError(
                f"Padding type {padding_type} is not implemented."
            )

        conv_block += [
            torch.nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
            norm_layer(dim),
            torch.nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [torch.nn.Dropout(0.5)]

        if padding_type == "reflect":
            conv_block += [torch.nn.ReflectionPad2d(1)]

        conv_block += [
            torch.nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
            norm_layer(dim),
        ]
        return torch.nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)  # Add skip connection


# Define the ResNet Generator
class ResnetGenerator(torch.nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=6):
        super(ResnetGenerator, self).__init__()
        model = [
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            torch.nn.BatchNorm2d(ngf),
            torch.nn.ReLU(True),
        ]

        # Downsampling
        model += [
            torch.nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(ngf * 2),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(ngf * 4),
            torch.nn.ReLU(True),
        ]

        # ResNet blocks
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * 4)]

        # Upsampling
        model += [
            torch.nn.ConvTranspose2d(
                ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            torch.nn.BatchNorm2d(ngf * 2),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(
                ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            torch.nn.BatchNorm2d(ngf),
            torch.nn.ReLU(True),
        ]

        # Final convolution
        model += [
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            torch.nn.Tanh(),
        ]

        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# Load the model
@app.on_event("startup")
def load_model():
    global model, device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResnetGenerator(
        input_nc=INPUT_NC, output_nc=OUTPUT_NC, ngf=NGF, n_blocks=N_BLOCKS
    ).to(device)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded successfully from {MODEL_PATH}")


@app.post("/deblur/")
async def deblur_image(file: UploadFile = File(...)):
    """Endpoint to deblur an input image."""
    try:
        # Load and preprocess the input image
        image = Image.open(file.file).convert("RGB")
        image = ImageOps.exif_transpose(image)  # Fix orientation based on EXIF

        # Extract the original file name without extension
        original_filename = os.path.splitext(file.filename)[0]
        deblurred_filename = f"deblurred_{original_filename}.png"

        # Ensure transformations match the dataset
        transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),  # Resize to match training dataset
                transforms.ToTensor(),  # Convert to tensor
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # Normalize to [-1, 1]
            ]
        )
        input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # Denormalize for visualization
        denormalize = transforms.Normalize(
            mean=[-1, -1, -1], std=[2, 2, 2]
        )  # Invert the normalization
        output_tensor = denormalize(output_tensor.squeeze(0).cpu())
        output_image = transforms.ToPILImage()(output_tensor)

        # Save the deblurred image
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, deblurred_filename)
        output_image.save(output_path)

        return {"message": "Image deblurred successfully", "output_path": output_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


# @app.post("/deblur/")
# async def deblur_image(file: UploadFile = File(...)):
#     """Endpoint to deblur an input image."""
#     try:
#         # Load and preprocess the input image
#         image = Image.open(file.file).convert("RGB")
#         image = ImageOps.exif_transpose(image)  # Fix orientation based on EXIF

#         # Ensure transformations match the dataset
#         transform = transforms.Compose(
#             [
#                 transforms.Resize((128, 128)),  # Resize to match training dataset
#                 transforms.ToTensor(),  # Convert to tensor
#                 transforms.Normalize(
#                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
#                 ),  # Normalize to [-1, 1]
#             ]
#         )
#         input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

#         # Perform inference
#         with torch.no_grad():
#             output_tensor = model(input_tensor)

#         # Denormalize for visualization
#         denormalize = transforms.Normalize(
#             mean=[-1, -1, -1], std=[2, 2, 2]
#         )  # Invert the normalization
#         output_tensor = denormalize(output_tensor.squeeze(0).cpu())
#         output_image = transforms.ToPILImage()(output_tensor)

#         # Save the deblurred image
#         os.makedirs(OUTPUT_DIR, exist_ok=True)
#         output_path = os.path.join(OUTPUT_DIR, "deblurred_image.png")
#         output_image.save(output_path)

#         return {"message": "Image deblurred successfully", "output_path": output_path}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "Inference service is running."}
