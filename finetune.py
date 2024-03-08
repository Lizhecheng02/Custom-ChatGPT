import os
import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
response = client.files.create(
    file=open("train_data.jsonl", "rb"),
    purpose="fine-tune"
)
print(response)
print("The file id for fine-tuning is:", response.id)

job = client.fine_tuning.jobs.create(
    training_file=response.id,
    model="gpt-3.5-turbo"
)
print(job)
print("The job id for fine-tuning is:", job.id)
