import os
import random
import asyncio
from typing import Annotated
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import kernel_function

load_dotenv()

class WeatherPlugin:
    """A simple weather plugin for demonstrating tool calling."""
    
    @kernel_function(
        name="get_weather",
        description="Get the current weather for a given location"
    )
    def get_weather(
        self, 
        location: Annotated[str, "The city or location to get weather for"]
    ) -> str:
        """Get the current weather for a given location.
        
        Args:
            location: The city or location to get weather for
            
        Returns:
            A string describing the current weather conditions
        """
        # Mock weather data - in a real application, this would call a weather API
        weather_conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "stormy", "foggy"]
        temperatures = list(range(15, 35))  # Temperature range in Celsius
        humidity_levels = list(range(30, 90))  # Humidity percentage
        
        condition = random.choice(weather_conditions)
        temperature = random.choice(temperatures)
        humidity = random.choice(humidity_levels)
        
        return (f"The weather in {location} is currently {condition} with a temperature of "
                f"{temperature}°C and humidity at {humidity}%.")

async def main():
    """Main function demonstrating automatic tool calling with Azure OpenAI."""
    try:
        service_id = "azure_openai"
        
        # Initialize the Azure OpenAI chat completion service
        chat_completion = AzureChatCompletion(
            service_id = service_id,
            deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            endpoint = os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version = os.environ["AZURE_OPENAI_API_VERSION"],
        )

        # Create execution settings with automatic tool calling enabled
        execution_settings = AzureChatPromptExecutionSettings(service_id = service_id, function_choice_behavior = FunctionChoiceBehavior.Auto(auto_invoke = True))

        # Initialize the kernel and add the weather plugin
        kernel = Kernel()
        weather_plugin = WeatherPlugin()
        kernel.add_plugin(weather_plugin, "weather")
        
        print("="*50)
        print("Example 1: Single city weather")
        print("="*50)
        
        # Example 1: Single city
        user_input = "What's the weather like in Sydney, Australia?"
        print(f"User: {user_input}")
        
        chat_history = ChatHistory()
        chat_history.add_system_message(
            "You are a helpful assistant that can check weather information. "
            "Use the available tools to answer user questions about weather."
        )
        chat_history.add_user_message(user_input)
        
        # Get response with automatic tool calling
        result = await chat_completion.get_chat_message_contents(
            chat_history=chat_history,
            settings=execution_settings,
            kernel=kernel
        )
        
        print(f"\nFinal Answer: {result[0].content}")
        
        print("\n" + "="*50)
        print("Example 2: Multiple cities")
        print("="*50)
        
        # Example 2: Multiple cities
        user_input = "Can you check the weather in Tokyo and London?"
        print(f"User: {user_input}")
        
        chat_history = ChatHistory()
        chat_history.add_system_message(
            "You are a helpful assistant that can check weather information. "
            "Use the available tools to answer user questions about weather."
        )
        chat_history.add_user_message(user_input)
        
        result = await chat_completion.get_chat_message_contents(
            chat_history=chat_history,
            settings=execution_settings,
            kernel=kernel
        )
        
        print(f"\nFinal Answer: {result[0].content}")
        
        print("\n" + "="*50)
        print("Example 3: No tool needed")
        print("="*50)
        
        # Example 3: Question that doesn't require tools
        user_input = "What's the capital of France?"
        print(f"User: {user_input}")
        
        chat_history = ChatHistory()
        chat_history.add_system_message(
            "You are a helpful assistant that can check weather information. "
            "Use the available tools to answer user questions about weather."
        )
        chat_history.add_user_message(user_input)
        
        result = await chat_completion.get_chat_message_contents(
            chat_history=chat_history,
            settings=execution_settings,
            kernel=kernel
        )
        
        print(f"\nFinal Answer: {result[0].content}")
            
    except KeyError as e:
        print(f"Error: Missing environment variable {e}")
        print("Please ensure your .env file contains:")
        print("- AZURE_OPENAI_ENDPOINT")
        print("- AZURE_OPENAI_DEPLOYMENT_NAME") 
        print("- AZURE_OPENAI_API_VERSION")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
