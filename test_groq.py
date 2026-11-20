
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("API key not found. Check your .env file.")
    raise SystemExit(1)

try:
    client = Groq(api_key=api_key)

    print("Testing Groq connection...", flush=True)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: Groq connection successful"
            }
        ],
        max_tokens=30,
    )

    print("SUCCESS:", response.choices[0].message.content)

except Exception as error:
    print("ERROR:", type(error).__name__, str(error))
