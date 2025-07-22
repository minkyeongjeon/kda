import os
import torch
import torch.nn as nn
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM
import pandas as pd
import argparse
import json
import logging
from peft import LoraConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

def infer(input_text: str, model, tokenizer, max_length: int = 128):
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='saltlux/Ko-Llama3-Luxia-8B')
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--name', type=str,default='')
    parser.add_argument('--checkpoint_path', type=str,default='')
    parser.add_argument('--data_path', type=str,default='')
    parser.add_argument('--lang', type=str, default='kr', choices=['en', 'kr'], help='Language for prompt: en or kr')
    args = parser.parse_args()
    CHECKPOINT_PATH = args.checkpoint_path

    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_PATH,device_map="auto")
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
    args.model_name,
    padding_side="right",
    model_max_length=128,
    )


    sample_path = args.data_path
    if '.json' in sample_path:
        with open(sample_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        filtered_data = [v for k, v in json_data.items() if v["Target"] == 1]
        sample_list = [d["Comment"] for d in filtered_data]
    elif '.csv' in sample_path:
        list_data_csv = pd.read_csv(sample_path) 
        sample_list = list_data_csv['implicit']
    else:
        raise ValueError("Invaide dataset format")
        
    logging.info(f"Total samples: {len(sample_list)}")

    model.eval()

    final_answer_lst = []
    for idx,sample in tqdm(enumerate(sample_list)):
        if args.lang == 'kr':
            input_text = f"""
                    Input에 성별, 지역, 종교, 정치 비하 slur, slang 표현이 있다면 이를 찾아 예쁜 말로 순화해주세요. 과정은 출력하지 말고,output은 하나의 문장으로만 출력해주세요.\n\nInput(순화할 문장):\n{sample}\n\nOutput(순화한 문장):"""
        elif args.lang == 'en':
            input_text = f"""
                    If the input contains slurs or slang that are derogatory toward gender, region, religion, or political views, please identify and rephrase them into more appropriate and respectful language. Do not output the reasoning process—just return a single, rephrased sentence as the output.\n\nInput (sentence to rephrase):\n{sample}\n\nOutput (rephrased sentence):"""
        output_text = infer(input_text, model, tokenizer)
        logging.debug(f"Output: {output_text}")
        answer = output_text.split("Output(순화한 문장):" if args.lang == 'kr' else "Output (rephrased sentence):")[-1]
        final_answer = pd.DataFrame({"Input":sample, "Output":answer},index=[idx])
        final_answer_lst.append(final_answer)
    
    final_csv = pd.concat(final_answer_lst)
    checkpoint_idx = CHECKPOINT_PATH.split("/")[-1].split("-")[-1]
    os.makedirs('./detoxification', exist_ok=True)
    final_csv.to_csv(f"./detoxification/{args.name}{checkpoint_idx}.csv",encoding='utf-8-sig')

if __name__ == "__main__":
    main()