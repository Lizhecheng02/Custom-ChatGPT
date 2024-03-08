import json
import pandas as pd
from datasets import load_dataset

DEFAULT_SYSTEM_PROMPT = "You are a teaching assistant for Ubuntu System. You should help the user to answer his question."


def create_dataset(question, answer):
    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


if __name__ == "__main__":
    df = load_dataset("mugithi/ubuntu_question_answer")
    print(df)
    train_data = df["train"].to_pandas().sample(100)
    train_data.reset_index(drop=True, inplace=True)
    print("The shape of train_data is:", train_data.shape)
    with open("train_data.jsonl", "w") as f:
        for idx, row in train_data.iterrows():
            data = json.dumps(create_dataset(row["question"], row["answer"]))
            f.write(data + "\n")
    print("Successfully Create Training Dataset!")
