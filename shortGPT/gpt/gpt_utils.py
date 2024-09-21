import json
import os
import re
from time import sleep, time
import requests
import yaml
from pathlib import Path

from shortGPT.config.api_db import ApiKeyManager


def num_tokens_from_messages(texts, model="chatgpt"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "chatgpt":  # note: future models may deviate from this
        if isinstance(texts, str):
            texts = [texts]
        score = 0
        for text in texts:
            score += 4 + len(encoding.encode(text))
        return score
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
        See https://github.com/openai/openai-python/blob/main/chatml.md for information""")


def extract_biggest_json(string):
    json_regex = r"\{(?:[^{}]|(?R))*\}"
    json_objects = re.findall(json_regex, string)
    if json_objects:
        return max(json_objects, key=len)
    return None


def get_first_number(string):
    pattern = r'\b(0|[1-9]|10)\b'
    match = re.search(pattern, string)
    if match:
        return int(match.group())
    else:
        return None


def load_yaml_file(file_path: str) -> dict:
    """Reads and returns the contents of a YAML file as a dictionary"""
    return yaml.safe_load(open_file(file_path))


def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data


def load_local_yaml_prompt(file_path):
    _here = Path(__file__).parent
    _absolute_path = (_here / '..' / file_path).resolve()
    json_template = load_yaml_file(str(_absolute_path))
    return json_template['chat_prompt'], json_template['system_prompt']


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def gpt3Turbo_completion(chat_prompt="", system="You are an AI that can give the answer to anything", temp=0.7, model="chatgpt", max_tokens=1000, remove_nl=True, conversation=None):
    api_key = ApiKeyManager.get_api_key("OPENAI")
    endpoint = "https://gs-openai-eastuss.openai.azure.com"
    max_retry = 5
    retry = 0
    while True:
        try:
            if conversation:
                messages = conversation
            else:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": chat_prompt}
                ]
            
            response = requests.post(
                f"{endpoint}/openai/deployments/{model}/chat/completions?api-version=2024-08-01-preview",
                headers={
                    "Content-Type": "application/json",
                    "api-key": api_key,
                },
                json={
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temp,
                    "top_p": 1.0  # Using top_p = 1.0 for deterministic completions
                }
            )

            response_data = response.json()
            text = response_data['choices'][0]['message']['content'].strip()

            if remove_nl:
                text = re.sub('\s+', ' ', text)

            filename = '%s_gpt3.txt' % time()
            log_dir = '.logs/gpt_logs'

            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            with open(f'{log_dir}/{filename}', 'w', encoding='utf-8') as outfile:
                outfile.write(f"System prompt: ===\n{system}\n===\n" +
                              f"Chat prompt: ===\n{chat_prompt}\n===\n" +
                              f'RESPONSE:\n====\n{text}\n===\n')

            return text

        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                raise Exception(f"Azure OpenAI error: {oops}")
            print(f'Error communicating with Azure OpenAI: {oops}')
            sleep(1)
