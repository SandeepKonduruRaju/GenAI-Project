from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()

print("AI Chatbot with Memory (type 'exit' to quit)\n")

# Conversation history
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."}
]
messages = messages[-10:]
while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # add user message to history
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=150
    )

    ai_reply = response.choices[0].message.content

    print("AI:", ai_reply)

    # add AI response to history
    messages.append({"role": "assistant", "content": ai_reply})