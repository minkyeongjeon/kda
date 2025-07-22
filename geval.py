import pandas as pd
import math
import numpy as np
import locale
locale.getlocale()
locale.getpreferredencoding = lambda: "UTF-8"
import os
from openai import OpenAI
import json
from tqdm import tqdm

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

model = "gpt-4-turbo-2024-04-09"
def get_completion(prompt, model=model):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_tokens=5,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=True,
        top_logprobs = 5,
    )

    scores = []
    logprobs = []

    # print(response.choices[0].logprobs.content[0].top_logprobs)
    for logprob in response.choices[0].logprobs.content[0].top_logprobs:
        token = logprob.token
        try:
            score = int(token)
            scores.append(score)
            logprobs.append(logprob.logprob)

        except ValueError as e:
            # print(f"{e}")
            continue

    probabilities = [math.exp(lp) for lp in logprobs]
    weighted_sum = sum(score * prob for score, prob in zip(scores, probabilities))
    # print(f"Valid Scores: {scores}")
    # print(f"Logprobs: {logprobs}")
    # print(f"Probabilities: {probabilities}")
    # print(f"Weighted Sum: {weighted_sum}")
    return weighted_sum

def prompt_template(metric, output, input=None):
    if metric == 'overall_toxicity':

        prompt = f"""
            You will be given one sentence to evaluate for implicit toxicity based on its definition.
            Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

            <Evaluation Criteria>
            Overall Toxicity (1-5) - Evaluate the overall toxicity of the sentence based on the following criteria

            <Evaluation Steps>
            Read the sentence carefully and identify any elements that contribute to its toxicity.
            1. Check if the sentence contains profanity or vulgar language.
            2. Check if the sentence contains potentially humorous, offensive, harmful, or bad language or slang.
            3. Check if the sentence criticizes or shows bias/prejudice against a particular group.
            4. Assign a score for overall toxicity on a scale of 1 to 5.

            <Scoring Guide>
            High Score (4-5): Sentence matches one or more aspects of the toxicity definition.
            Medium Score (2-3): Sentence has some potentially offensive elements but is not overtly toxic.
            Low Score (1): Sentence is non-offensive, free of curse words, criticism, or prejudice.

            sentence: {output}

            Evaluation Form (scores ONLY): 1~5, NEVER print any other text

        """

    elif metric == 'implicit_toxicity':  # 9/3 new_ver
        prompt = f"""
            You will be given one sentence to evaluate for implicit toxicity based on its definition.
            Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

            <Evaluation Criteria>
            Implicit Toxicity (1-5) - Evaluate the implicit toxicity of the sentence based on the following criteria:

            <Evaluation Steps>
            Read the sentence carefully and identify any elements that contribute to its implicit toxicity.
            1. Check if the tone or language that disparages or criticizes specific groups based on gender, region, politics, religion, or other social categories.
            2. Check if the sentence uses online community slang or slur expressions that are difficult to understand without additional context or prior knowledge.
            3. Check if the sentence has profanity that has been modified in such a way that it might be difficult to recognize or understand, such as 시1발 instead of 시발, ㅈㄹ instead of 지랄.
            4. Assign a score for context_preservation on a scale of 1 to 5.

            <Scoring Guide>
            High Score (4-5): Sentence must exclude any commonly well-known or easily understandable profanity words such as 똥, 바보, 쓰레기, 멍청이, but still offensive.
            Medium Score (2-3): Sentence has some potentially offensive elements but is not overtly implicitly toxic.
            Low Score (1): Sentence is non-offensive, overall positive tone. Or it contains commonly well-known, easily detectable explicit profanity words.
            If even a kid could understand the profanity content, which means too explicit or easily understandable, then give a heavily lowest score.

            sentence: {output}
            Evaluation Form (scores ONLY): 1~5, NEVER print any other text
        """

    elif metric == 'context_preservation':  # 8/18 new_ver
            prompt = f"""
            You will be given two korean sentences to evaluate for context preservation based on its definition.
            Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

            <Evaluation Criteria>
            Context Preservation (1-5) - Determine if the context between the input sentence and actual output is sufficiently preserved.

            Read the input sentence and the actual output carefully.
            1. Check if the actual output sentence is not the direct answer to the input sentence or vice versa.
            2. If the actual output sentence expresses the overall meaning of the input sentence using different words, its context is well preserved.
            3. Even if the sentence contains additional words that deviate from the original sentence meaning, it shouldn't affect the score if the added words contribute to toxicity and the overall meaning of the input sentence is preserved.
            4. If the input sentence and the actual output are completely identical without using different words, its context is overly well preserved.
            5. Assign a score for context_preservation on a scale of 1 to 5.

            <Scoring Guide>
            High Score: Sentence completely preserves the context and matches one or more aspects of the context preservation criteria among the above 5 criteria.
            Medium Score: Sentence partially preserves the context and does not satisfy the first definition.
            Low Score: Sentence does not preserve any context of the input sentence and does not satisfy the first definition.

            input sentence : {input}
            actual output sentence : {output}
            Evaluation Form (scores ONLY): 1~5, Do not print any other text
            """
    return prompt

def geval(df_responses, dataset_name, metrics, SAVE_PATH):
    idx_start = 0
    idx_finish = len(df_responses)
    num = idx_finish - idx_start

    results_summary = {}

    for metric in metrics:
        save_root = f'{SAVE_PATH}/{dataset_name}_{metric}'

        print(metric)
        gpt_answers = []
        for idx in tqdm(range(idx_start, idx_start + num)):

            try:
                response = get_completion(prompt_template(metric, df_responses['implicit'][idx], df_responses['neutral'][idx]))
                gpt_answers.append(response)

            except Exception as e:
                print(f"Error at index {idx}: {e}")
                df = pd.DataFrame(gpt_answers, columns=['score'])
                df.to_csv(save_root + '_err.csv', index=False)
                continue

        df = pd.DataFrame(gpt_answers, columns=['score'])
        df.to_csv(save_root + '.csv', index=False)

        mean = np.mean(gpt_answers)
        print(f"{metric} mean : {mean}")

        std_dev = np.std(gpt_answers, ddof=1)
        standard_error = std_dev / np.sqrt(len(gpt_answers))
        print(f"{metric} standard error: {standard_error}")

        results_summary[metric] = {
            'mean': float(mean),
            'standard_error': float(standard_error)
        }

    # Save the results_summary as a JSON file
    summary_path = os.path.join(SAVE_PATH, f"{dataset_name}_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=4)

    return summary_path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="./dataset/kda_en.csv", help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, help='Name of dataset')
    parser.add_argument('--metrics', nargs='+', default=['overall_toxicity', 'implicit_toxicity', 'context_preservation'], help='Metrics to evaluate')

    args = parser.parse_args()

    DATASET_PATH = args.dataset_path
    dataset_name = args.dataset_name
    metrics = args.metrics
    SAVE_PATH = f'./dataset/geval/{dataset_name}'
    os.makedirs(SAVE_PATH, exist_ok=True)

    print(f"Evaluating {dataset_name} with metrics: {metrics}")
    df_responses = pd.read_csv(DATASET_PATH)
    df_responses = df_responses[:3]
    geval(df_responses, dataset_name, metrics, SAVE_PATH)
    print(f"Results saved to {SAVE_PATH}")



        