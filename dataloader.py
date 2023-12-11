import pandas as pd
import os
import json
import torch
from typing import Optional, Union
import numpy as np
from datasets import Dataset
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from deep_translator import GoogleTranslator
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score
from preprocessing import data_df

VER=2
# TRAIN WITH SUBSET OF 60K
NUM_TRAIN_SAMPLES = 1_024
# PARAMETER EFFICIENT FINE TUNING
# PEFT REQUIRES 1XP100 GPU NOT 2XT4
USE_PEFT = False
# NUMBER OF LAYERS TO FREEZE
# DEBERTA LARGE HAS TOTAL OF 24 LAYERS
FREEZE_LAYERS = 18 # 16, 14, 12, 10, 8, 0
# BOOLEAN TO FREEZE EMBEDDINGS
FREEZE_EMBEDDINGS = True
# LENGTH OF CONTEXT PLUS QUESTION ANSWER
MAX_INPUT = 256
# HUGGING FACE MODEL
MODEL = 'microsoft/deberta-v3-large' # 'deepset/deberta-v3-large-squad2' 'OpenAssistant/reward-model-deberta-v3-large-v2' # 'microsoft/deberta-v3-large'



option_to_index = {option: idx for idx, option in enumerate('ABCD')}
index_to_option = {v: k for k,v in option_to_index.items()}

def preprocess(example):
    first_sentence = [ "[CLS] " + example['explanation'] ] * 4
    second_sentences = [" #### " + example['question'] + " [SEP] " + example[option] + " [SEP]" for option in 'ABCD']
    tokenized_example = tokenizer(first_sentence, second_sentences, truncation=True,
                                  max_length=MAX_INPUT, add_special_tokens=False)
    tokenized_example['label'] = option_to_index[example['answer']]

    return tokenized_example

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch


######
from sklearn.model_selection import train_test_split

df_train, df_valid = train_test_split(data_df, test_size=0.1, random_state=42)

print(df_train.shape, df_valid.shape)


######
tokenizer = AutoTokenizer.from_pretrained(MODEL)
dataset_valid = Dataset.from_pandas(df_valid)
dataset = Dataset.from_pandas(df_train)
dataset = dataset.remove_columns(["__index_level_0__"])
dataset


######
tokenized_dataset_valid = dataset_valid.map(preprocess, remove_columns=['question', 'explanation', 'A', 'B', 'C', 'D', 'answer'])
tokenized_dataset = dataset.map(preprocess, remove_columns=['question', 'explanation', 'A', 'B', 'C', 'D', 'answer'])
tokenized_dataset