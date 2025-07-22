# K/DA: Automated Data Generation Pipeline for Detoxifying Implicitly Offensive Language in Korean (ACL 2025)

[Project Page](https://minkyeongjeon.github.io/kda_projectpage/) 
[ArXiv](https://arxiv.org/abs/2506.13513)


## Overview
⚠️ Caution: This research includes content that might be considered offensive.

`Implicit offensiveness` is a form of offensive language characterized by a tone of disregard or mockery that conveys derogatory meaning, such as sarcasm or social bias within context, while avoiding explicit expressions. 

- disregard and mockery, consistent with past definitions of implicit offensiveness

- community-specific slang that is familiar within certain groups but difficult for outsiders to interpret

- variations of profanity used to avoid detection

Specifically, communities with **high-context languages** such as Korean are more likely to use these types of implicit offensive expressions. 

Therefore, we use these categories to guide the data generation process. Furthermore, we demonstrate the **language- and model- agnostic** nature of this pipeline by generating data in English.

<img width="2601" height="1136" alt="Image" src="https://github.com/user-attachments/assets/21351a98-c4fa-47ff-8a21-7c4bbde89ca4" />
An overview of K/DA, the pipeline for automated offensive language data generation.


**Step 1**

Retrieve 9 semantically similar sentences from the community using cosine similarity. An LLM then **synthesizes a toxic version** by incorporating trend-aligned slang from these sentences.

**Step 2**

An off-the-shelf LLM **filters** the candidates based on two criteria:

• `Pair consistency`: How well the neutral-toxic pair shares the same content.

• `Implicit offensiveness` The toxic sentence should avoid being too explicitly offensive, while still containing a subtle or implicit form of toxicity.

---

## Environment Setup
```
export OPENAI_API_KEY='YOUR API KEY'
```
```
pip install -r requirements.txt
```

## Synthesizing implicit toxic dataset
1. Run `gen_toxic.ipynb` to generate candidate toxic samples.
2. Run `filtering.ipynb` to filter for implicit hate expressions.

## Evaluation (GEval)
```
cd kda
python geval.py --dataset_path "./dataset/kda_en.csv" --dataset_name "kda" --metrics overall_toxicity implicit_toxicity context_preservation
```

### Instruction Tuning
Although the prompt in this code is optimized for Korean, it can still serve as a guideline for detoxifying hate speech in other languages.

**Training**
```
python ./instruction_tuning.py \
  --model_name "YOUR_MODEL_NAME" \
  --max_epochs NUM_EPOCHS \
  --name "YOUR_RUNNING_NAME" \
  --repo_name "YOUR_HF_REPO" \
  --data_path "./dataset/kda_en.csv"
```

**Inference**
```
cd kda
python ./inference.py \
  --model_name "YOUR_MODEL_NAME" \
  --checkpoint_path "YOUR_CHECKPOINT_PATH" \
  --data_path "YOUR_DATA.csv" \
  --lang en \
  --name "infer_DATA_en"
```

## Dataset
❗️Please fill out [this form](https://forms.gle/WDECRuuH328jw93L6) before using K/DA dataset.

The dataset is also available on [HuggingFace](https://huggingface.co/datasets/minkyeongjeon/kda-dataset)

---

## Citation
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
