import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchtext import datasets
from tqdm import tqdm
from itertools import islice

from impl.utils import *
from impl.mlp_models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_model = True  # Set to False to only evaluate the model
model_path = "./data/text_model.pt"

# You might need more epochs with higher batch sizes to converge
# Increasing the depth of the model might let it converge earlier
training_epochs = 2
training_batch_size = 50
training_batch_count = 100  # Change this
eval_token_count = 200

# Input is a 64-bit index. Output is a probability vector over possible byte values
model_input_width = 64
model = NormalMLP(
    in_dim=model_input_width,
    out_dim=255,
    hidden_width=256,
    hidden_depth=16,
    is_residual=True,
).to(device)

# Trial and error <3. Feel free to change these parameters.
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=5000, gamma=0.9)
lossFunc = nn.CrossEntropyLoss()

text_dataset = datasets.CC100("./data/torchtext/", language_code="en")
text_loader = DataLoader(text_dataset, shuffle=False, batch_size=training_batch_size)


def train():
    maxTrainedIndex = 0

    model.train()
    for _ in tqdm(range(training_epochs)):
        batch_it = tqdm(
            islice(text_loader, training_batch_count),
            total=training_batch_count,
            leave=False,
        )

        for batch_idx, (batchValues, _) in enumerate(batch_it):
            optimizer.zero_grad()

            batchValues = batchValues.to(device).reshape(-1, 1)

            beginIndex = batch_idx * training_batch_size
            endIndex = beginIndex + training_batch_size - 1
            maxTrainedIndex = max(maxTrainedIndex, endIndex)

            inputs = binary_arange(beginIndex, endIndex + 1, model_input_width, device)

            # Otherwise the loss function does it and it creates a copy. [i, j] -> [i]
            output = model(inputs).reshape(-1)

            loss = lossFunc(output, batchValues)
            loss.backward()
            optimizer.step()
            scheduler.step()

    torch.save(model.state_dict(), model_path)
    print(f"Training finished. Model saved. Highest trained index: {maxTrainedIndex}")


def evaluate():
    inputs = binary_arange(0, eval_token_count, model_input_width, device)
    output = torch.argmax(torch.softmax(model(inputs)))

    for token in output:
        print(chr(token))


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
