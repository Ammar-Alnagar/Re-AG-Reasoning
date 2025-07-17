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
        # You can add more documents here as needed.
    ]

    # Set up the client to use the Hugging Face Inference API.
    # Replace `YOUR_HUGGINGFACE_TOKEN` with your actual token.
    async with ReagClient(
        model="huggingface/gpt-neo-2.7B",  # For example, use GPT-Neo 2.7B or any available model.
        api_base="https://api-inference.huggingface.co",  # Hugging Face endpoint.
        headers={"Authorization": "Bearer YOUR_HUGGINGFACE_TOKEN"},
    ) as client:
        print("Chat session with Hugging Face model. Type 'exit' to quit.")
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

