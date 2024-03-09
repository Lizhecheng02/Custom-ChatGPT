import json
from datasets import load_dataset
import argparse
from tqdm import trange

DEFAULT_SYSTEM_PROMPT = "You are a teaching assistant for Ubuntu System. You should help the user to answer his question."


def create_entry(question, answer):
    return json.dumps({
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    })


def generate(sample_size):
    # Load the dataset
    df = load_dataset("mugithi/ubuntu_question_answer")
    print(df)

    # Sample the dataset
    train_data = df["train"].to_pandas().sample(sample_size, random_state=42)
    train_data.reset_index(drop=True, inplace=True)
    print("The shape of train_data is:", train_data.shape)

    # Create the training dataset
    jsonified_dataset = train_data.apply(lambda x: create_entry(x["question"], x["answer"]), axis=1).str.cat(sep="\n")
    with open("train_data.jsonl", "w") as f:
        f.write(jsonified_dataset)
    print("Successfully Create Training Dataset!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Training Dataset")
    parser.add_argument("-s", "--sample_size", type=int, default=100, help="The size of the training dataset")
    args = parser.parse_args()

    generate(args.sample_size)
