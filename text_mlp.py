import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from impl.utils import *
from impl.mlp_models import *
from impl.text_dataset import *

device = get_torch_device()

train_model = True  # Set to False to only evaluate the model
model_path = "./data/text_model.pt"

# Input is a 64-bit index. Output is a probability vector over possible byte values
model_input_width = 64
model = NormalMLP(
    in_dim=model_input_width,
    out_dim=256,
    hidden_width=1024,
    hidden_depth=512,
    is_residual=True
).to(device)

# Trial and error <3. Feel free to change these parameters.
training_epochs = 2
training_batch_size = 1000
optimizer = optim.SGD(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=5000, gamma=0.9)
lossFunc = nn.CrossEntropyLoss()

textDataset = TextModelDataset()
textLoader = DataLoader(textDataset, shuffle=False, batch_size=training_batch_size)


def train():
    model.train()
    for _ in tqdm(range(training_epochs)):
        for batch_idx, batchValues in enumerate(tqdm(textLoader, leave=False)):
            optimizer.zero_grad()

            batchValues = batchValues.to(device)

            beginIndex = batch_idx * len(batchValues)
            endIndex = beginIndex + len(batchValues) - 1

            inputs = binary_arange(beginIndex, endIndex + 1, model_input_width, device)
            output = model(inputs).reshape(-1, 256)

            loss = lossFunc(output, batchValues)
            loss.backward()
            optimizer.step()
            scheduler.step()

    torch.save(model.state_dict(), model_path)
    print(f"Training finished. Model saved.")


def evalAndMetaTrain(optimizerArg, modelInput):
    optimizerArg.zero_grad()
    output = torch.softmax(model(modelInput), dim=0)
    sampledTensor = torch.multinomial(output, num_samples=1)
    # CrossEntropyLoss expects batched inputs
    loss = lossFunc(output.reshape(-1, 256), sampledTensor)
    loss.backward()
    optimizerArg.step()
    return sampledTensor


def evaluate():
    model.train()
    evalOptimizer = optim.SGD(model.parameters(), lr=1e-4)
    evalOffset = len(textDataset)
    evalTokenCount = 100

    for idx in range(evalOffset, evalOffset + evalTokenCount):
        encodedIdx = binary_encode_tensor(
            torch.tensor(idx).to(device), model_input_width
        )
        sampledTensor = evalAndMetaTrain(evalOptimizer, encodedIdx)
        print(chr(sampledTensor.item()), end="")


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
