import pytorch_lightning as pl
import torch
from transformers import T5ForConditionalGeneration, AdamW
from utils import (
    gen_input_text,
    USE_SYNTAX,
    MODEL_NAME,
    INPUT_MAX_LEN,
    OUTPUT_MAX_LEN,
)
import random

TRAIN_BATCH_SIZE = 8 # batch size of training
VAL_BATCH_SIZE = 8 # batch size for validation

class TextRuleT5Dataset:
    def __init__(self, rule_list, tokenizer):
        self.rule_list = rule_list
        self.tokenizer = tokenizer
        self.input_max_len = INPUT_MAX_LEN
        self.output_max_len = OUTPUT_MAX_LEN

    def __len__(self):
        return len(self.rule_list)

    def __getitem__(self, idx):
        text, rules, support = gen_input_text(
            self.rule_list,
            idx,
            get_rules = True,
            use_syntax_path_tokens = USE_SYNTAX,
            train = True,
        )

        if idx % 1000 == 0:
            print("Text: ", text)
            print("Rules to match: ", rules)

        input_encoded = self.tokenizer(
            text,
            max_length = self.input_max_len,
            add_special_tokens = True,
            padding = "max_length",
            truncation = True,
            return_attention_mask = True,
            return_tensors = "pt"
        )

        labels = self.tokenizer(
            rules,
            max_length = self.output_max_len,
            add_special_tokens = True,
            padding = "max_length",
            truncation = True,
            return_attention_mask = True,
            return_tensors = "pt"
        )

        return {
            "input_ids": input_encoded["input_ids"].flatten(),
            "attention_mask": input_encoded["attention_mask"].flatten(),
            "labels": labels["input_ids"].flatten(),
        }

class TextRuleT5DataLoad(pl.LightningDataModule):
    def __init__(self, train_data, val_data, tokenizer):
        super().__init__()
        self.train_data_raw = train_data
        self.val_data_raw = val_data
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        self.train_data = TextRuleT5Dataset(self.train_data_raw, self.tokenizer)
        self.val_data = TextRuleT5Dataset(self.val_data_raw, self.tokenizer)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
          self.train_data,
          batch_size=TRAIN_BATCH_SIZE,
          shuffle=True,
          num_workers=2
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
          self.val_data,
          batch_size=VAL_BATCH_SIZE,
          num_workers=2
        )

class TextRuleT5Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)
