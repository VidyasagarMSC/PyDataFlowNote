import os
from langfuse.openai import openai
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()
langfuse = Langfuse(
	public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
	secret_key=os.getenv("LANGFUSE_SECRET_KEY")
)

trace = langfuse.trace(
	name="calculator"
)

openai.langfuse_debug = True

output = openai.chat.completions.create(
	model="gpt-4o-mini",
	messages=[
		{"role": "system","content": "You are a very accurate calculator. You output only the result of the calculation."},
		{"role": "user", "content": "1 + 1 = "}],
	name="test-chat",
	metadata={"someMetadataKey": "someValue"},
).choices[0].message.content

print(output)
