import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from random import randint

from impl.utils import *
from impl.mlp_models import *
from impl.text_dataset import *

device = get_torch_device()

train_model = True  # Set to False to only evaluate the model
model_path = "./data/text_model.pt"

training_epochs = 2
training_batch_size = 1000

# Input is a 64-bit index. Output is a probability vector over possible byte values
model_input_width = 64
model = InvertedBottleneckMLP(
    in_dim=model_input_width,
    out_dim=256,
    hidden_width=256,
    hidden_depth=32,
).to(device)

# Trial and error <3. Feel free to change these parameters.
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=5000, gamma=0.9)
lossFunc = nn.CrossEntropyLoss()

textDataset = TextModelDataset()
textLoader = DataLoader(textDataset, shuffle=False, batch_size=training_batch_size)

def train():
    highestTrainedIndex = 0

    model.train()
    for _ in tqdm(range(training_epochs)):
        for batch_idx, batchValues in enumerate(tqdm(textLoader, leave=False)):
            optimizer.zero_grad()

            batchValues = batchValues.to(device)

            beginIndex = batch_idx * len(batchValues)
            endIndex = beginIndex + len(batchValues) - 1
            highestTrainedIndex = max(highestTrainedIndex, endIndex)

            inputs = binary_arange(beginIndex, endIndex + 1, model_input_width, device)
            output = model(inputs).reshape(-1, 256)
            
            loss = lossFunc(output, batchValues)
            loss.backward()
            optimizer.step()
            scheduler.step()

    torch.save(model.state_dict(), model_path)
    print(f"Training finished. Model saved. Highest trained index: {highestTrainedIndex}")


def evaluate():
    model.eval()
    # TODO: change this?
    evalOffset = randint(0, len(textDataset)-5000)
    evalTokenCount = 5000
    
    inputs = binary_arange(evalOffset, evalOffset +  evalTokenCount, model_input_width, device)
    output = torch.softmax(model(inputs), dim=1)
    samples = torch.multinomial(output, num_samples=1)

    for token in samples:
        print(chr(token.item()), end='')


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
