import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from impl.utils import *
from impl.mlp_models import *
from impl.text_dataset import *

device = get_torch_device()

train_model = True  # Set to False to only evaluate the model
model_path = "./data/text_model.pt"
dataset_path = "./data/test.txt"

# Input is a 64-bit index. Output is a probability vector over possible byte values
model_input_width = 64
model = ResMLPModel(
    in_dim=model_input_width, out_dim=256, hidden_width=4096, hidden_depth=16
).to(device)

# Trial and error <3. Feel free to change these parameters.
training_epochs = 2
textDataset = BinaryIndexedTextDataset(filename=dataset_path, bits=model_input_width)
textLoader = DataLoader(
    textDataset,
    shuffle=False,
    batch_size=4096,
    num_workers=4,
    pin_memory=True,
    pin_memory_device=device,
)

optimizer = optim.SGD(model.parameters(), lr=1e-3)
scaler = GradScaler()
loss_fn = nn.CrossEntropyLoss()

# 10 scheduler steps per epoch, last LR will be LR*gamma^10
scheduler = StepLR(
    optimizer, gamma=0.9, step_size=(len(textDataset) // textLoader.batch_size // 10)
)


def train():
    model.train()
    for _ in tqdm(range(training_epochs)):
        for inputs, targets in tqdm(textLoader, leave=False):
            optimizer.zero_grad()

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Runs the forward pass with autocasting.
            with torch.autocast(device_type=device, dtype=torch.float16):
                output = model(inputs)
                loss = loss_fn(output, targets)

            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

    torch.save(model.state_dict(), model_path)
    print(f"Training finished. Model saved.")


def evalAndMetaTrain(optimizerArg, modelInput):
    optimizerArg.zero_grad()
    output = torch.softmax(model(modelInput), dim=0)
    sampledTensor = torch.multinomial(output, num_samples=1)
    # CrossEntropyLoss expects batched inputs, [[0]] instead of [0]
    # but we are evaluating a single token at once
    loss = loss_fn(output.unsqueeze(0), sampledTensor)
    loss.backward()
    optimizerArg.step()
    return sampledTensor


def evaluate():
    print("Evaluating model...")
    model.train()
    evalOptimizer = optim.SGD(model.parameters(), lr=1e-4)
    evalOffset = len(textDataset)
    evalTokenCount = 100

    for idx in range(evalOffset, evalOffset + evalTokenCount):
        encodedIdx = binary_encode_tensor(
            torch.tensor(idx, device=device), model_input_width
        )
        sampledTensor = evalAndMetaTrain(evalOptimizer, encodedIdx)
        print(chr(sampledTensor.item()), end="")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
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
