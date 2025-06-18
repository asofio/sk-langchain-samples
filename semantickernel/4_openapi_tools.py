import asyncio
import requests
import os

from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from semantic_kernel.filters import AutoFunctionInvocationContext, FilterTypes

load_dotenv()

async def main():
    """OpenAPI Sample Client"""
    kernel = Kernel()

    @kernel.filter(FilterTypes.AUTO_FUNCTION_INVOCATION)
    async def auto_function_invocation_filter(context: AutoFunctionInvocationContext, next):
        """A filter that will be called for each function call in the response."""
        print(f"\nFunction: {context.function.name}")

        function_calls = context.chat_history.messages[-1].items
        print(f"Number of function calls: {len(function_calls)}")
        for function_call in function_calls:
            print(f"Function name: {function_call.name}")
            print(f"Arguments: {function_call.arguments}")
        
        # if we don't call next, it will skip this function, and go to the next one
        await next(context)

    service_id = "azure_openai"
    
    # Initialize the Azure OpenAI chat completion service
    chat_completion = AzureChatCompletion(
        service_id = service_id,
        deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version = os.environ["AZURE_OPENAI_API_VERSION"],
    )

    # Load OpenAPI spec from the remote URL
    openapi_url = "https://sofio-calculator-plugin.azurewebsites.net/openapi.json"
    response = requests.get(openapi_url)
    response.raise_for_status()
    openapi_data = response.json()

    kernel.add_plugin_from_openapi(plugin_name="openApiPlugin", openapi_parsed_spec=openapi_data)

    execution_settings = AzureChatPromptExecutionSettings(service_id = service_id, function_choice_behavior = FunctionChoiceBehavior.Auto(auto_invoke = True))

    user_input = "What is 2+3? Take the result and multiply it by 4."
    print(f"\nUser: {user_input}\n")
    
    chat_history = ChatHistory()
    chat_history.add_system_message(
        "You are a helpful assistant that can perform arithmetic. "
    )
    chat_history.add_user_message(user_input)
    
    # Get response with automatic tool calling
    result = await chat_completion.get_chat_message_contents(
        chat_history=chat_history,
        settings=execution_settings,
        kernel=kernel
    )
    
    print(f"\nFinal Answer: {result[0].content}")


if __name__ == "__main__":
    asyncio.run(main())

