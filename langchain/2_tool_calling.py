import os
import random
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

@tool
def get_weather(location: str) -> str:
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

def main():
    """Main function demonstrating tool calling with Azure OpenAI."""
    try:
        # Initialize the LLM with Azure OpenAI
        model = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0.1,  # Lower temperature for more consistent responses
        )
        
        # Bind the tool to the model
        model_with_tools = model.bind_tools([get_weather])
        
        # Create a message asking about weather
        messages = [
            HumanMessage(content="What's the weather like in Sydney, Australia?")
        ]
        
        print("User: What's the weather like in Sydney, Australia?")
        print("Assistant: Let me check the weather for you...")
        
        # Get the model's response with tool calls
        response = model_with_tools.invoke(messages)
        
        # Check if the model wants to use tools
        if response.tool_calls:
            print(f"\nModel is calling tool: {response.tool_calls[0]['name']}")
            print(f"With arguments: {response.tool_calls[0]['args']}")
            
            # Execute the tool call
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'get_weather':
                    location = tool_call['args']['location']
                    weather_result = get_weather(location)
                    print(f"\nWeather result: {weather_result}")
        else:
            # If no tool calls, just print the response
            print(f"\nAssistant: {response.content}")
            
        print("\n" + "="*50)
        print("Example 2: Multiple cities")
        print("="*50)
        
        # Example with multiple cities
        messages = [
            HumanMessage(content="Can you check the weather in Tokyo and London?")
        ]
        
        print("User: Can you check the weather in Tokyo and London?")
        response = model_with_tools.invoke(messages)
        
        if response.tool_calls:
            print(f"\nModel is making {len(response.tool_calls)} tool call(s):")
            
            for i, tool_call in enumerate(response.tool_calls, 1):
                if tool_call['name'] == 'get_weather':
                    location = tool_call['args']['location']
                    weather_result = get_weather(location)
                    print(f"{i}. {weather_result}")
        else:
            print(f"\nAssistant: {response.content}")
            
    except KeyError as e:
        print(f"Error: Missing environment variable {e}")
        print("Please ensure your .env file contains:")
        print("- AZURE_OPENAI_ENDPOINT")
        print("- AZURE_OPENAI_DEPLOYMENT_NAME") 
        print("- AZURE_OPENAI_API_VERSION")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
