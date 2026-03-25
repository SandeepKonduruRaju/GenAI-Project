from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()

print("AI Chatbot with Memory Control (type 'exit' to quit)\n")

messages = [
    {
        "role": "system",
        "content": "You are an expert AI tutor. Explain concepts step-by-step like teaching a beginner. Use simple language, examples, and bullet points and make sure toeo explain within 150 tokens. Always ask if the user wants to learn more or has any questions after your explanation."
    }
]

MAX_MESSAGES = 10   # memory window

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Add user message
    messages.append({"role": "user", "content": user_input})

    # Keep only last N messages (IMPORTANT)
    if len(messages) > MAX_MESSAGES:
        system_message = messages[0]
        messages = [system_message] + messages[-(MAX_MESSAGES-1):]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=150
    )

    ai_reply = response.choices[0].message.content

    print("AI:", ai_reply)

    # Add AI response
    messages.append({"role": "assistant", "content": ai_reply})