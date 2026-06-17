from llmsays import llmsays
from dotenv import load_dotenv
import os
load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY")
NIVIDIA_API_KEY=os.getenv("NIVIDIA_API_KEY")
BASETEN_API_KEY=os.getenv("BASETEN_API_KEY")
FIREWORKSAI_API_KEY=os.getenv("FIREWORKSAI_API_KEY")
response = llmsays(
    "Write a code only to implement a function in Go that calculates the factorial of a number and nothing else explanation is required",
    max_tokens=1024,
)
print(response)