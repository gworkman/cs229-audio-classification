import torch
from torch import nn
import torchmetrics
import torch.utils.data
import torchvision
import torchaudio
import tone_dataset
import pytorch_lightning as pl
from pathlib import Path
import matplotlib.pyplot as plt


class ToneClassifier(pl.LightningModule):

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(18496, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 4),
            nn.Softmax(1)
        )

        for layer in self.model.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform(layer.weight)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        accuracy = (torch.argmax(y_hat, 1) == y).float().mean()
        self.log('training_loss', loss)
        self.log('training_accuracy', accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        accuracy = (torch.argmax(y_hat, 1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


audio_dir = Path('../data/preprocessed')

transforms = torchvision.transforms.Compose([
    torchaudio.transforms.TimeMasking(20, p=0.8),
    torchaudio.transforms.FrequencyMasking(20),
    torchvision.transforms.RandomCrop((128, 128))
])

ds = tone_dataset.ToneDataset(audio_dir / 'spectrogram.pkl', transform=transforms)
train_len = int(len(ds) * 0.8)
test_len = (len(ds) - train_len) // 2
validation_len = int(len(ds) - train_len - test_len)

train_ds, test_ds, validation_ds = torch.utils.data.random_split(ds, [train_len, test_len, validation_len])

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)
validation_dl = torch.utils.data.DataLoader(validation_ds, batch_size=32, shuffle=False)


model = ToneClassifier()

trainer = pl.Trainer(max_epochs=10, log_every_n_steps=5)
trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=validation_dl)
