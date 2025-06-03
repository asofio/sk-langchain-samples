import os
import asyncio
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory

load_dotenv()

async def main():
    # Initialize the kernel
    kernel = Kernel()
    
    # Add Azure OpenAI chat completion service
    service_id = "azure_openai"
    kernel.add_service(
        AzureChatCompletion(
            service_id=service_id,
            deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
    )
    
    # Get the chat completion service
    chat_completion = kernel.get_service(type=AzureChatCompletion, service_id=service_id)
    
    # Create chat history and add user message
    chat_history = ChatHistory()
    chat_history.add_user_message("Tell me three facts about Kangaroos.")
    
    # Stream the response
    print("Streaming response:")
    async for chunk in chat_completion.get_streaming_chat_message_contents(
        chat_history=chat_history,
        settings=chat_completion.get_prompt_execution_settings_class()(
            service_id=service_id
        )
    ):
        for message in chunk:
            if message.content:
                print(message.content, end="")
    
    print("\n")  # Add newline at the end

if __name__ == "__main__":
    asyncio.run(main())
