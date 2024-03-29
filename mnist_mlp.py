import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from impl.utils import *
from impl.mlp_models import *

device = get_torch_device()

train_model = True  # Set to False to only evaluate the model
model_path = "./data/mnist_model.pt"
output_path = "./data/out.png"

# MNIST dataset is 28x28
training_image_dim = (28, 28)

# You might need more epochs with higher batch sizes to converge
# Increasing the depth of the model lets it converge earlier
training_epochs = 800
training_batch_size = 50

# Input is a 38-bit index. Output is a luminance value between [0, 1]
model_input_width = 38
model = nn.Sequential(ResMLPModel(
            in_dim=model_input_width,
            out_dim=1,
            hidden_width=128,
            hidden_depth=4
), nn.Sigmoid()).to(device)

# Trial and error <3. Feel free to change these parameters.
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=50, gamma=0.9)
lossFunc = nn.MSELoss()

mnist_dataset = datasets.MNIST(
    "./data/mnist", train=True, download=True, transform=transforms.ToTensor()
)
mnist_loader = DataLoader(mnist_dataset, shuffle=False, batch_size=training_batch_size)


def train():
    model.train()
    for _ in tqdm(range(training_epochs)):
        for batch_idx, (batchValues, _) in enumerate(tqdm(mnist_loader, leave=False)):
            optimizer.zero_grad()

            batchValues = batchValues.to(device).reshape(-1)
            pixelCountInBatch = (
                len(batchValues) * training_image_dim[0] * training_image_dim[1]
            )

            beginIndex = batch_idx * pixelCountInBatch
            endIndex = beginIndex + pixelCountInBatch - 1

            inputs = binary_arange(beginIndex, endIndex + 1, model_input_width, device)

            # Otherwise the loss function does it and it creates a copy. [i, j] -> [i]
            output = model(inputs).reshape(-1)

            loss = lossFunc(output, batchValues)
            loss.backward()
            optimizer.step()

        scheduler.step()

    torch.save(model.state_dict(), model_path)
    print(f"Training finished. Model saved.")


def evaluate():
    pixelCount = training_image_dim[0] * training_image_dim[1]
    imageCount = 200
    images = []

    model.eval()
    for i in tqdm(range(imageCount)):
        luminance_map = torch.Tensor()
        with torch.no_grad():
            beginIndex = i * pixelCount
            inputs = binary_arange(
                beginIndex, beginIndex + pixelCount, model_input_width, device
            )
            output = model(inputs)
            luminance_map = output.reshape(
                1, training_image_dim[0], training_image_dim[1]
            )

        tensor_to_pil = transforms.ToPILImage()(luminance_map)
        images.append(tensor_to_pil)

    canvas = pillow_generate_canvas(images, math.ceil(math.sqrt(imageCount)))
    canvas.save(output_path)
    print("Evaluation complete. Image file saved.")


if train_model:
    try:
        model.load_state_dict(torch.load(model_path))
        print("Loaded existing model for training.")
    except:
        print("Loading existing model failed. Training a new one...")
    train()
else:
    model.load_state_dict(torch.load(model_path))

evaluate()
