import torch
import torchvision.utils as vutils
import os
import argparse


# Define the Generator model
class Generator(torch.nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(ngf * 8),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 4),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 2),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            torch.nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


# Parse arguments
parser = argparse.ArgumentParser(description="DCGAN Test Script")
parser.add_argument(
    "--model-path", type=str, required=True, help="Path to the trained model"
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="./output",
    help="Directory to save generated images",
)
parser.add_argument(
    "--num-images", type=int, default=10, help="Number of images to generate"
)
parser.add_argument("--nz", type=int, default=100, help="Size of the latent vector")
parser.add_argument("--ngf", type=int, default=64, help="Feature maps in generator")
parser.add_argument(
    "--nc", type=int, default=3, help="Number of channels in the generated images"
)
parser.add_argument(
    "--seed", type=int, default=None, help="Random seed for reproducibility"
)
args = parser.parse_args()

# Set the seed for reproducibility
if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# Check for device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the trained generator model
model = Generator(nz=args.nz, ngf=args.ngf, nc=args.nc).to(device)
if not os.path.exists(args.model_path):
    raise FileNotFoundError(f"Model file not found at {args.model_path}")

model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()
print(f"Model loaded successfully from {args.model_path}")

# Generate random latent vectors
noise = torch.randn(args.num_images, args.nz, 1, 1, device=device)

# Generate images
with torch.no_grad():
    generated_images = model(noise)

# Save generated images
os.makedirs(args.output_dir, exist_ok=True)
output_path = os.path.join(args.output_dir, "generated_samples.png")
vutils.save_image(generated_images, output_path, normalize=True)
print(f"Generated images saved to {output_path}")
