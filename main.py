# Python STD
import argparse
from collections import defaultdict
import torch

# Dataset
from datasets import load_metric
from dataset import Xsum_Dataset

# Model
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AdamW
# Trainer

from train import Trainer
# Logging
from torch.utils.tensorboard import SummaryWriter

from settings import training_setting


# ########################### Global Settings ###########################
device ="cuda" if torch.cuda.is_available() else "cpu"


parser =  argparse.ArgumentParser(description=" A parser for T5 summarizer")
parser.add_argument("--tr_batch_size", type=str, default="5")
parser.add_argument("--val_batch_size", type=str, default="5")
parser.add_argument("--epoch", type=str, default="1")

parser.add_argument("--lr", type=str, default="0.001")
parser.add_argument("--metric", type=str, default="rouge")


# Extract commandline arguments   
args = parser.parse_args()

tr_batch_size = int(args.tr_batch_size) if args.tr_batch_size!=None else  training_setting["tr_batch_size"]
val_batch_size = int(args.val_batch_size) if args.val_batch_size!=None else  training_setting["val_batch_size"]
test_batch_size = training_setting["test_batch_size"]
epoch = int(args.epoch) if args.epoch!=None else  training_setting["epoch"]
lr = float(args.lr) if args.lr!=None else  training_setting["lr"]
metric = args.metric if args.metric!=None else  training_setting["metric"]


# Tensorboard Summary Writer
tb_logger = SummaryWriter("logs",flush_secs=5)
# Rouge Metric
metric = load_metric("rouge")

# ########################### Load Dataset ###########################
xsum_train_data = Xsum_Dataset (split = 'train', source_length= 512, target_length=200)
xsum_val_data = Xsum_Dataset(split='validation', source_length=512, target_length=200)
xsum_test_data = Xsum_Dataset(split='test', source_length=512, target_length=200)

# Build DataLoader

xsum_train_loader = torch.utils.data.DataLoader( dataset= xsum_train_data, batch_size= tr_batch_size)
xsum_val_loader = torch.utils.data.DataLoader( dataset= xsum_val_data, batch_size= val_batch_size)
xsum_test_loader = torch.utils.data.DataLoader( dataset= xsum_test_data, batch_size= test_batch_size)

# ########################### Model Settings ###########################

model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
optimizer = AdamW(model.parameters(), lr= lr)

# ########################### Training Settings ###########################

# Tokenizer
wikitokenizer = T5Tokenizer.from_pretrained("t5-small")

# Trainer
trainer = Trainer(xsum_train_loader, xsum_val_loader, model, wikitokenizer, optimizer, metric) 


if __name__ =="__main__":
    train_stats = defaultdict(list)
    for ep in range(epoch):
        train_stats = trainer.train(train_stats)


