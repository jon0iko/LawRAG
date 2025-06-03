import os
from openai import OpenAI
from openai.types.chat import ChatCompletionChunk
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

prompt = "What is the meaning of life?"

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
    stream=True,
    max_tokens=100,
    temperature=0
)

# stream response to the terminal
for item in response:
    chunk: ChatCompletionChunk = item
    # Check if choices list is not empty, delta exists, and content is not None
    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
print()