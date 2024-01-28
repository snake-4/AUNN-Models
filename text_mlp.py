import time
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
model_out_width = 256
model_hidden_width = 4096
model_hidden_depth = 16

# Trial and error <3. Feel free to change these parameters.
training_epochs = 2
training_save_interval = 1800
training_batch_size = 8192
training_dataloader_workers = 4

textDataset = BinaryIndexedTextDataset(filename=dataset_path, bits=model_input_width)
loss_fn = nn.CrossEntropyLoss()


def train(model):
    textLoader = DataLoader(
        textDataset,
        batch_size=training_batch_size,
        shuffle=False,
        num_workers=training_dataloader_workers,
        pin_memory=True,
        pin_memory_device=device,
    )
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    scaler = GradScaler()

    # 10 scheduler steps per epoch, last LR will be LR*gamma^10
    scheduler = StepLR(
        optimizer,
        gamma=0.9,
        step_size=(len(textDataset) // textLoader.batch_size // 10),
    )

    last_save = time.monotonic()
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

            time_now = time.monotonic()
            if time_now > last_save + training_save_interval:
                last_save = time_now
                tqdm.write(f"Model saved. time.monotonic()={time_now}")
                torch.save(model.state_dict(), model_path)

    torch.save(model.state_dict(), model_path)
    print("Training finished. Model saved.")


def evalAndMetaTrain(model, optimizerArg, modelInput):
    optimizerArg.zero_grad()
    output = torch.softmax(model(modelInput), dim=0)
    sampledTensor = torch.multinomial(output, num_samples=1)
    # CrossEntropyLoss expects batched inputs, [[0]] instead of [0]
    # but we are evaluating a single token at once
    loss = loss_fn(output.unsqueeze(0), sampledTensor)
    loss.backward()
    optimizerArg.step()
    return sampledTensor


def evaluate(model, start_index, length):
    print("Evaluating model...")
    model.train()
    evalOptimizer = optim.SGD(model.parameters(), lr=1e-4)

    for idx in range(start_index, start_index + length):
        encodedIdx = binary_encode_tensor(
            torch.tensor(idx, device=device), model_input_width
        )
        sampledTensor = evalAndMetaTrain(model, evalOptimizer, encodedIdx)
        print(chr(sampledTensor.item()), end="")


def main():
    model = ResMLPModel(
        in_dim=model_input_width,
        out_dim=model_out_width,
        hidden_width=model_hidden_width,
        hidden_depth=model_hidden_depth,
    ).to(device)

    if train_model:
        try:
            model.load_state_dict(torch.load(model_path))
            print("Loaded existing model for training.")
        except:
            print("Loading existing model failed. Training a new one...")
        train(model)
    else:
        model.load_state_dict(torch.load(model_path))
    evaluate(model, start_index=len(textDataset), length=100)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
