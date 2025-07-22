# K/DA Project Page

[🔗 Project Page](https://minkyeongjeon.github.io/kda_projectpage/) for **K/DA**: Automated Data Generation Pipeline for Detoxifying Implicitly Offensive Language in Korean (ACL 2025)  

## Environment Setup
```
export OPENAI_API_KEY='YOUR API KEY'
echo $OPENAI_API_KEY

pip install -r requirements.txt
```

## Synthesizing implicit toxic dataset
1. Run `gen_toxic.ipynb` to generate candidate toxic samples.
2. Run `filtering.ipynb` to filter for implicit hate expressions.

## GEval
```
cd kda
python geval.py --dataset_path "./dataset/kda_en.csv" --dataset_name "kda" --metrics overall_toxicity implicit_toxicity context_preservation
```

### Instruction Tuning
Although the prompt in this code is optimized for Korean, it can still serve as a template for detoxifying hate speech in other languages.

Training
```
python ./instruction_tuning.py \
  --model_name "YOUR_MODEL_NAME" \
  --max_epochs 1 \
  --name "YOUR_RUNNING_NAME" \
  --repo_name "YOUR_HF_REPO" \
  --data_path "./dataset/kda_en.csv"
```

Inference
Detoxify using our instruction tuned model
```
kda
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

## 📁 Dataset & Insturction-tuned model
❗️Please fill out [this form](https://forms.gle/WDECRuuH328jw93L6) before using K/DA dataset.

Access the dataset and instruction-tuned model on [HuggingFace](https://huggingface.co/datasets/minkyeongjeon/kda-dataset)

---

## 📌 Citation
```
  <pre><code>@misc{jeon2025kdaautomateddatageneration,
  title={K/DA: Automated Data Generation Pipeline for Detoxifying Implicitly Offensive Language in Korean}, 
  author={Minkyeong Jeon and Hyemin Jeong and Yerang Kim and Jiyoung Kim and Jae Hyeon Cho and Byung-Jun Lee},
  year={2025},
  eprint={2506.13513},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2506.13513}
```

---