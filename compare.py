import os
import pandas as pd
import openai
from openai import OpenAI
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
new_model_id = "ft:gpt-3.5-turbo-0125:personal::90HshtNN"


def compare_models(question):
    completion1 = client.chat.completions.create(
        model=new_model_id,
        messages=[
            {"role": "system", "content": "You are a teaching assistant for Ubuntu System. You should help the user to answer his question."},
            {"role": "user", "content": question}
        ]
    )
    new_output = completion1.choices[0].message.content

    completion2 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a teaching assistant for Ubuntu System. You should help the user to answer his question."},
            {"role": "user", "content": question}
        ]
    )
    original_output = completion2.choices[0].message.content

    return new_output, original_output


if __name__ == "__main__":
    df = load_dataset("mugithi/ubuntu_question_answer")
    test_data = df["test"].to_pandas().sample(10)
    test_data.reset_index(drop=True, inplace=True)
    print("The shape of test_data is:", test_data.shape)

    results = pd.DataFrame(
        columns=[
            "id", "question", "correct answer",
            "new output", "original output"
        ]
    )
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
        new_output, original_output = compare_models(row["question"])
        new_row = pd.DataFrame({
            'id': [idx],
            'question': [row["question"]],
            'correct answer': [row["answer"]],
            'new output': [new_output],
            'original output': [original_output]
        })
        results = pd.concat([results, new_row], ignore_index=True)
        results.to_csv("results.csv", index=False)
        results.to_json("results.json", orient="records")

    print("Run successfully!")
