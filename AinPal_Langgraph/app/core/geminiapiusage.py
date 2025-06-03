import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Convert to SecretStr if not None
api_key_secret = SecretStr(api_key) if api_key is not None else None
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key_secret)

prompt = "You are a helpful assistant. Answer the following question: What is the capital of France?"

try:
  response = model.invoke(prompt)
except Exception as error:
  print(f"Error invoking model: {error}")