import os
import argparse
import logging
from dataclasses import dataclass
from typing import Dict, Sequence
import copy

import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.optim import Adam
from torch.utils.data import Dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

IGNORE_INDEX = -100

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class ITDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, lang: str = 'kr', verbose=False):
        super(ITDataset, self).__init__()

        # load data
        logging.warning("Loading data...")
        if '.csv' in data_path:
            dataset = pd.read_csv(data_path)
            if 'implicit' not in dataset.columns or 'neutral' not in dataset.columns:
                raise ValueError("Dataset must contain 'implicit' and 'neutral' columns")
            logging.warning(f"Dataset columns: {dataset.columns}")
            implicit_sentences = dataset['implicit'].tolist()
            neutral_sentences = dataset['neutral'].tolist()
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

        logging.warning(f"Dataset len: {len(implicit_sentences)}, implicit: {implicit_sentences[0]}, neutral: {neutral_sentences[0]}")

        sources = []
        for example in implicit_sentences:
            if example != "":
                if lang == 'kr':
                    prompt = f"""
                        Input에 성별, 지역, 종교, 정치 비하 slur, slang 표현이 있다면 이를 찾아 예쁜 말로 순화해주세요. 과정은 출력하지 말고,output은 하나의 문장으로만 출력해주세요.\n\nInput(순화할 문장):\n{example}\n\nOutput(순화한 문장):"""
                elif lang == 'en':
                    prompt = f"""
                        If the input contains slurs or slang that are derogatory toward gender, region, religion, or political views, please identify and rephrase them into more appropriate and respectful language. Do not output the reasoning process—just return a single, rephrased sentence as the output.\n\nInput (sentence to rephrase):\n{example}\n\nOutput (rephrased sentence):"""
                else:
                    raise ValueError(f"Unsupported language: {lang}")
            else:
                raise Exception("Error")
            sources.append(prompt)

        targets = []
        for example in neutral_sentences:
            targets.append(f"{example}{tokenizer.eos_token}" if isinstance(example, str) else f"{tokenizer.eos_token}")

        examples = [s + t for s, t in zip(sources, targets)]

        # source data tokenized
        sources_tokenized = self._tokenize_fn(sources, tokenizer)
        examples_tokenized = self._tokenize_fn(examples, tokenizer)

        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX

        data_dict = dict(input_ids=input_ids, labels=labels)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        logging.warning("Dataset loaded. len: %d"%(len(self.labels)))

    def _tokenize_fn(self, strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
        return dict(
            input_ids=input_ids,
            input_ids_lens=input_ids_lens,
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForITDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='saltlux/Ko-Llama3-Luxia-8B')
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--repo_name', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--lang', type=str, default='kr', choices=['en', 'kr'], help='Language for prompt: en or kr')
    args = parser.parse_args()

    epochs = args.max_epochs
    cur_name = args.name

    quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
      args.model_name,
      quantization_config=quant_config,
      device_map="auto"
  )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    )
    model.add_adapter(peft_params, adapter_name="adapter_1")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
    args.model_name,
    padding_side="right",
    model_max_length=512,
    )


    train_dataset = ITDataset(data_path=args.data_path, tokenizer=tokenizer, lang=args.lang)
    eval_dataset = None
    data_collator = DataCollatorForITDataset(tokenizer=tokenizer)

    # Check the first sample
    logging.warning("Dataset len", len(train_dataset))
    logging.warning('input : %s' % train_dataset.input_ids[0])
    logging.warning('output: %s' % train_dataset.labels[0])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.warning(device)

    OUTPUT_DIR = F'./outputs_{cur_name}'

    if not os.path.exists(OUTPUT_DIR) : os.makedirs(OUTPUT_DIR)

    logging.warning(f"Current exp name : {cur_name}")

    training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_steps = 3,
    save_steps=2500,
    warmup_steps=5,
    prediction_loss_only=True,
    learning_rate=2e-4,
    run_name=cur_name
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.save_state()

    token = os.environ.get("HUGGINGFACE_TOKEN")
    if token is None:
        raise RuntimeError("HUGGINGFACE_TOKEN environment variable not set.")
    model.push_to_hub(f'{args.repo_name}/{cur_name}', use_auth_token=token)

if __name__ == "__main__":
    main()

