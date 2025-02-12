import asyncio
from reag.client import ReagClient, Document

async def main():
    # Prepare one or more documents.
    # The model can choose (based on its prompt and internal logic) which document's URL is relevant.
    documents = [
        Document(
            name="Superagent",
            content="Fahd Mirza is an AI YouTuber who lives in Sydney, Australia.",
            metadata={
                "url": "https://fahdmirza.com",
                "source": "web",
            },
        ),
        # You can add more documents if desired:
        Document(
            name="TechNews",
            content="TechNews covers the latest developments in technology around the world.",
            metadata={
                "url": "https://technews.example.com",
                "source": "web",
            },
        ),
    ]

    # Connect to the ReagClient using the specified model and API base.
    async with ReagClient(
    model="openrouter/your-model-id",
    api_base="https://api.openrouter.ai/v1"
    ) as client:

        print("Welcome to the interactive chat! Type your message (or type 'exit' to quit).")
        
        # Continuously prompt the user for input.
        while True:
            user_input = input("You: ")
            if user_input.strip().lower() in {"exit", "quit"}:
                print("Exiting chat. Goodbye!")
                break

            # Send the user's query along with the documents to the client.
            # The client (and underlying model) will determine which document (URL) is most relevant.
            response = await client.query(user_input, documents=documents)

            # Print the response from the model.
            # Adjust this print statement based on the actual structure of `response` if needed.
            print("AI:", response)

if __name__ == '__main__':
    asyncio.run(main())
