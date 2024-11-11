from transformers import AutoTokenizer
import pytorch_lightning as pl
import torch
import json, random
from model import TextRuleT5DataLoad, TextRuleT5Model
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils import RULE_TYPE, MODEL_NAME, MODEL_SAVE_PATH, SUFFIX
import warnings

# TRAIN_PATH = f"../nyt29/train_upto_5_anchor_{str(RULE_TYPE)}_rules{SUFFIX}.jsonl"
# DEV_PATH = f"../nyt29/dev_upto_5_anchor_{str(RULE_TYPE)}_rules{SUFFIX}.jsonl"

# TRAIN_PATH = f"/workspace/combined/anchor_word/anchor_word_{str(RULE_TYPE)}_rules_upto_5_train.jsonl"
# DEV_PATH = f"/workspace/combined/anchor_word/anchor_word_{str(RULE_TYPE)}_rules_upto_5_dev.jsonl"

TRAIN_PATH = f"../nyt29/train_upto_5_anchor_{str(RULE_TYPE)}_rules_pphrase_sep_entry.jsonl"
DEV_PATH = f"../nyt29/dev_upto_5_anchor_{str(RULE_TYPE)}_rules_pphrase_sep_entry.jsonl"

EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(train_rules, dev_rules, tokenizer):
    dataload = TextRuleT5DataLoad(train_rules, dev_rules, tokenizer)
    dataload.setup()
    device = DEVICE
    model = TextRuleT5Model()
    model.to(device)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )

    checkpoint = ModelCheckpoint(
        dirpath=".",
        filename=MODEL_SAVE_PATH,
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        callbacks=[early_stopping, checkpoint],
        max_epochs=EPOCHS,
        accelerator="auto",
    )

    trainer.fit(model, dataload)

torch.set_float32_matmul_precision('medium')
pl.seed_everything(100)
warnings.filterwarnings("ignore")

train_rules = []
print("Loading data: ", TRAIN_PATH)
with open(TRAIN_PATH, "r") as file:
    for line in file:
        json_obj = json.loads(line)
        train_rules.append(json_obj)

dev_rules = []
print("Loading data: ", DEV_PATH)
with open(DEV_PATH, "r") as file:
    for line in file:
        json_obj = json.loads(line)
        dev_rules.append(json_obj)

random.shuffle(train_rules)
print("Number of train instances: ", len(train_rules))
print("Number of dev instances: ", len(dev_rules))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train(train_rules, dev_rules, tokenizer)
