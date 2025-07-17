import asyncio
from reag.client import ReagClient, Document

async def main():
    # Define your document(s)
    documents = [
        Document(
            name="Superagent",
            content="Fahd Mirza is an AI YouTuber who lives in Sydney, Australia.",
            metadata={
                "url": "https://fahdmirza.com",
                "source": "web",
            },
        ),
        # Add more documents if desired.
    ]

    # Set up the client to use the Groq API.
    # Replace `your-model-id` and `YOUR_GROQ_TOKEN` with the appropriate values.
    async with ReagClient(
        model="groq/your-model-id",             # Update with your Groq model identifier.
        api_base="https://api.groq.com/v1",       # Update with the correct Groq endpoint.
        headers={"Authorization": "Bearer YOUR_GROQ_TOKEN"},
    ) as client:
        print("Chat session with Groq model. Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.strip().lower() in {"exit", "quit"}:
                print("Exiting chat. Goodbye!")
                break

            # Query the model along with your documents.
            response = await client.query(user_input, documents=documents)
            print("AI:", response)

if __name__ == '__main__':
    asyncio.run(main())
