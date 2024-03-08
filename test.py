import openai
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
status = client.fine_tuning.jobs.retrieve("ftjob-k66N8QIGu1ehgQmGRIbhXUuD")
print(status)

if status.status == "succeeded":
    new_model_id = status.fine_tuned_model
    print("The new model id is:", new_model_id)

completion = client.chat.completions.create(
    model=new_model_id,
    messages=[
        {"role": "system", "content": "You are a teaching assistant for Ubuntu System. You should help the user to answer his question."},
        {"role": "user", "content": "sudo apt-get install postgresql?"}
    ]
)
print(completion.choices[0].message.content)
