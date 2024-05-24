import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.trainer import Trainer
from src.base_model import BaseModel
from src.experimental import KANModel
from src.dataset import JSONLDataset

from datetime import datetime


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
dataset = ...
# Define the sizes of training, validation, and test sets
train_size = int(0.9 * len(dataset))  # 90% of the data for training
val_size = int(0.1 * len(dataset))   # 10% of the data for validation
batch_size = 128

# Split the dataset into training, validation, and test sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoader instances for each set
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset._collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=dataset._collate_fn)
Base_model = BaseModel(... , device=device)
KAN_model = KANModel(... , device=device)

trainer_base = Trainer(
model=Base_model,
optimizer=torch.optim.Adam(Base_model.parameters(), lr=0.0001),
oss_function=nn.CrossEntropyLoss(),
log_steps=100
)

trainer_KAN = Trainer(
model=KAN_model,
optimizer=torch.optim.Adam(KAN_model.parameters(), lr=0.0001),
#loss_function=nn.CrossEntropyLoss(),
log_steps=100
)

losses_base = trainer_base.train(train_loader, val_loader, epochs=20)
losses_KAN = trainer_KAN.train(train_loader, val_loader, epochs=20)
# Get the current date and time
current_date_time = datetime.now()

# Format the date as a string
current_date_str = current_date_time.strftime("%Y-%m-%d")

torch.save(Base_model.state_dict(), f"./saves/Base_model{current_date_str}.pth")
print("saved Base_model!")

torch.save(KAN_model.state_dict(), f"./saves/KAN_model{current_date_str}.pth")
print("saved KAN_model!") 
