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


class TextMLP:
    def __init__(self):
        self.device = get_torch_device()
        self.model = ResMLPModel(
            in_dim=model_input_width,
            out_dim=model_out_width,
            hidden_width=model_hidden_width,
            hidden_depth=model_hidden_depth,
        ).to(self.device)
        self.text_dataset = BinaryIndexedTextDataset(
            filename=dataset_path, bits=model_input_width
        )
        self.text_loader = DataLoader(
            self.text_dataset,
            batch_size=training_batch_size,
            shuffle=False,
            num_workers=training_dataloader_workers,
            pin_memory=True,
            pin_memory_device=self.device,
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.scaler = GradScaler()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3)
        # Per epoch, last LR will be LR*gamma^10
        self.scheduler = StepLR(
            self.optimizer,
            gamma=0.9,
            step_size=(len(self.text_dataset) // self.text_loader.batch_size // 10),
        )

        self.meta_learn_epoch = 200
        self.meta_optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        # Per meta learn call, last LR will be LR*gamma^10
        self.meta_scheduler = StepLR(
            self.meta_optimizer, 
            gamma=0.9,
            step_size=(self.meta_learn_epoch // 10)
        )

    def train(self):
        last_save = time.monotonic()
        self.model.train()
        for _ in tqdm(range(training_epochs)):
            for inputs, targets in tqdm(self.text_loader, leave=False):
                self.optimizer.zero_grad()

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Runs the forward pass with autocasting.
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    output = self.model(inputs)
                    loss = self.loss_fn(output, targets)

                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scheduler.step()
                self.scaler.update()

                time_now = time.monotonic()
                if time_now > last_save + training_save_interval:
                    last_save = time_now
                    self.save_model()
                    tqdm.write(f"Model saved. time.monotonic()={time_now}")

        self.save_model()
        print("Training finished. Model saved.")

    def evaluate(self, start_index, length):
        print("Evaluating model...")
        self.model.train()
        self.__meta_learn_string(start_index, "Hi, my name is ")

        for idx in range(start_index, start_index + length):
            encodedIdx = binary_encode_tensor(
                torch.tensor(idx, device=self.device), model_input_width
            )
            sampledTensor = self.__meta_evaluate_and_learn(encodedIdx)
            print(chr(sampledTensor.item()), end="", flush=True)

    def save_model(self):
        torch.save(self.model.state_dict(), model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(model_path))

    @property
    def highest_trained_index(self) -> int:
        # TODO: Fix this?
        return len(self.text_dataset)

    def __meta_learn_string(self, index: int, string: str):
        inputs = binary_arange(
            index, index + len(string), bits=model_input_width, device=self.device
        )
        targets = torch.tensor([ord(x) for x in string], device=self.device)
        self.__meta_learn_raw(inputs, targets)

    def __meta_learn_raw(
        self, modelInputs: torch.Tensor, targetOutputs: torch.Tensor
    ):
        for _ in range(self.meta_learn_epoch):
            self.meta_optimizer.zero_grad()
            output = torch.softmax(self.model(modelInputs), dim=0)
            loss = self.loss_fn(output, targetOutputs)
            loss.backward()
            self.meta_optimizer.step()
            self.meta_scheduler.step()

    def __meta_evaluate_and_learn(self, modelInput: torch.Tensor) -> torch.tensor:
        self.meta_optimizer.zero_grad()
        output = torch.softmax(self.model(modelInput), dim=0)
        sampledTensor = torch.multinomial(output, num_samples=1)
        # CrossEntropyLoss expects batched inputs, [[0]] instead of [0]
        # but we are evaluating a single token at once
        loss = self.loss_fn(output.unsqueeze(0), sampledTensor)
        loss.backward()
        self.meta_optimizer.step()
        return sampledTensor


def main():
    textMLP = TextMLP()

    if train_model:
        try:
            textMLP.load_model()
            print("Loaded existing model for training.")
        except:
            print("Loading existing model failed. Training a new one...")
        textMLP.train()
    else:
        textMLP.load_model()
    textMLP.evaluate(start_index=textMLP.highest_trained_index, length=100)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
