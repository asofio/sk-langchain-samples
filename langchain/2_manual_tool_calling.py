"""
Manual Tool Calling Example with Azure OpenAI and LangChain Core

1. Send user message to model with bound tools
2. Check if model wants to call tools
3. Execute tool functions manually using tool.invoke()
4. Send tool results back to model
5. Get final response
"""

import os
import random
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

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


def execute_tool_calls(user_input, tools, model_with_tools):
    """
    Execute tool calls from the model response and get the final answer.
    
    Args:
        response: The model response containing tool calls
        tools: List of available tools
        messages: Current message history
        model_with_tools: The model with tools bound to it
        
    Returns:
        Final response from the model after tool execution
    """
    messages = [HumanMessage(content=user_input)]
    response = model_with_tools.invoke(messages)
    
    if response.tool_calls:
        print(f"\n🔧 Model wants to call {len(response.tool_calls)} tool(s):")
        
        # Add the model's response to messages
        messages.append(response)
        
        # Execute each tool call manually
        for i, tool_call in enumerate(response.tool_calls, 1):
            print(f"  {i}. Calling tool: {tool_call['name']}")
            print(f"     Arguments: {tool_call['args']}")
            
            # Find the tool by name and invoke it properly
            for tool in tools:
                if tool.name == tool_call['name']:
                    tool_result = tool.invoke(tool_call['args'])

                    print(f"     Result: {tool_result}")
                    
                    # Create a tool message with the result
                    tool_message = ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call['id']
                    )
                    
                    messages.append(tool_message)
                    break
                else:
                    print(f"     Tool not found: {tool_call['name']}")
        
    else:
        print("No tool calls needed.")

    print(f"\n🤖 Getting response from model...")
    final_response = model_with_tools.invoke(messages)
    print(f"Answer: {final_response.content}")


def main():
    """Main function demonstrating manual tool calling with Azure OpenAI."""
    try:
        # Initialize the LLM with Azure OpenAI
        model = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0.1,  # Lower temperature for more consistent responses
        )
        
        # Create a list of tools
        tools = [get_weather]
        
        # Bind tools to the model
        model_with_tools = model.bind_tools(tools)
        
        # ==================================================================
        # Example 1: Single city
        print("="*50)
        print("Example 1: Single city weather (Manual Tool Calling)")
        print("="*50)
        
        user_input = "What's the weather like in Sydney, Australia?"
        print(f"User: {user_input}")
        
        execute_tool_calls(user_input, tools, model_with_tools)
    
        # ==================================================================
        # Example 2: Multiple cities
        print("\n" + "="*50)
        print("Example 2: Multiple cities (Manual Tool Calling)")
        print("="*50)
    
        user_input = "Can you check the weather in Tokyo and London?"
        print(f"User: {user_input}")        

        execute_tool_calls(user_input, tools, model_with_tools)
        
        # ==================================================================
        # Example 3: Question that doesn't require tools        
        print("\n" + "="*50)
        print("Example 3: No tool needed (Manual Check)")
        print("="*50)
        
        user_input = "What's the capital of France?"
        print(f"User: {user_input}")

        execute_tool_calls(user_input, tools, model_with_tools)

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
