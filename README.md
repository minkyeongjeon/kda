
Project page

## Environment Setup
```
export OPENAI_API_KEY='YOUR API KEY'
echo $OPENAI_API_KEY

pip install -r requirements.txt
```

## Synthesizing implicit toxic dataset
1. run `gen_toxic.ipynb`
2. run `filtering.ipynb`

## GEval
```
cd kda
python geval.py --dataset_path "./dataset/kda_en.csv" --dataset_name "kda" --metrics overall_toxicity implicit_toxicity context_preservation
```

## Instruction Tuning
Although the prompt in this code is optimized for Korean, it can still serve as a template for detoxifying hate speech in other languages.

### Training
```
python ./instruction_tuning.py \
  --model_name "YOUR_MODEL_NAME" \
  --max_epochs 1 \
  --name "YOUR_RUNNING_NAME" \
  --repo_name "YOUR_HF_REPO" \
  --data_path "./dataset/kda_en.csv"
```

### Inference
Detoxify using our instruction tuned model
```
cd kda
python ./inference.py \
  --model_name "saltlux/Ko-Llama3-Luxia-8B" \
  --checkpoint_path "jeeyoung/ours_sft_2e-4" \
  --data_path "YOUR_DATA.csv" \
  --lang kr \
  --name "infer_DATA_kr"
```

English
```
cd kda
python ./inference.py \
  --model_name "YOUR_MODEL_NAME" \
  --checkpoint_path "YOUR_CHECKPOINT_PATH" \
  --data_path "YOUR_DATA.csv" \
  --lang en \
  --name "infer_DATA_en"
```
