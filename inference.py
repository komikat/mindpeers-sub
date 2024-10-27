from sys import argv

import torch
from llama_cpp import Llama

from train import MultiLabelClassifier

invert_labels = {
    "Extracted Concern": {
        0: "feeling hopeful",
        1: "happy and excited",
        2: "constantly worried",
        3: "feeling very anxious",
        4: "feeling much better",
        5: "not eating properly",
        6: "worried about health",
        7: "confused about job prospects",
        8: "extremely stressed",
        9: "can't sleep well",
        10: "feeling very low",
    },
}

polarity_prompt = """You are a helpful assistant who has been trained to analyze user input and identify sentiment polarity. \
                     Answer every prompt only with a Positive, Negative or Neutral. Make sure you only answer with one word."""
category_prompt = """You are a helpful assistant who has been trained to analyze user input and identify sentiment category. \
                   Answer every prompt only with one of (Positive Outlook, Stress, Insomnia, Depression, Anxiety, Career Confusion, Health Anxiety, Eating Disorder). \
                   Example: user: "My mind feels like it’s worried about health."  answer: Health Anxiety\
                   Make sure you only answer with one of the given terms."""
intensity_prompt = """You are a helpful assistant who has been trained to analyze user input and identify sentiment intensity. \
                   Answer every prompt only with one of the first 10 numbers from 0 to 10. \
                   Example: user: "I am constantly worried these days."  answer: 7\
                   Make sure you only answer with one of the first 10 numbers."""


def init_models(filepath="best_model.pt"):
    model = MultiLabelClassifier()
    model.load_state_dict(torch.load(filepath, map_location=torch.device("cpu")))
    llm = Llama.from_pretrained(
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q6_K.gguf",
    )
    print("Succesfully loaded llama3.2 and concern classifier!")
    return llm, model


def run_model(llama, st, text):
    def create_completion(system_prompt):
        p = llama.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{text}"},
            ]
        )
        return p["choices"][0]["message"]["content"]

    p = create_completion(polarity_prompt)
    c = create_completion(category_prompt)
    i = create_completion(intensity_prompt)
    outputs = st(text)
    _e_c = int(torch.argmax(outputs["concern"]))
    e_c = invert_labels["Extracted Concern"][_e_c]

    return p, c, i, e_c


if __name__ == "__main__":
    if len(argv) < 2:
        print("Please enter some text")

    llm, st = init_models()
    print(run_model(llm, st, " ".join(argv[1:])))
