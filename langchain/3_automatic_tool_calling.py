"""
Automatic Tool Calling Example with Azure OpenAI and LangGraph

This example demonstrates how to use LangGraph's create_react_agent for automatic
tool calling, which provides a higher-level abstraction compared to manual tool calling.
The agent automatically handles the tool calling flow:

1. Receives user input
2. Decides which tools to call (if any)
3. Executes tools automatically
4. Provides final response

This example uses the same get_weather tool as the manual example for comparison.
"""

import os
import random
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

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


def execute_automatic_tool_calling(user_input, agent):
    """
    Execute automatic tool calling using LangGraph agent.
    
    Args:
        user_input: The user's question or request
        agent: The LangGraph react agent
        
    Returns:
        Agent's response after automatic tool execution
    """
    print(f"User: {user_input}")
    
    # The agent automatically handles tool calling, execution, and response generation
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})
    
    # Extract the final message from the agent's response
    final_message = response["messages"][-1]
    print(f"\nAgent Response: {final_message.content}")
    
    return final_message


def main():
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
        
        # Create a React agent using LangGraph - this handles all tool calling automatically
        agent = create_react_agent(model, tools)
        
        # ==================================================================
        # Example 1: Single city weather (Automatic Tool Calling)
        print("\nExample 1: Single city weather")
        print("-" * 40)
        
        user_input = "What's the weather like in Sydney, Australia?"
        execute_automatic_tool_calling(user_input, agent)
    
        # ==================================================================
        # Example 2: Multiple cities (Automatic Tool Calling)
        print("\n" + "="*60)
        print("\nExample 2: Multiple cities")
        print("-" * 40)
    
        user_input = "Can you check the weather in Tokyo and London?"
        execute_automatic_tool_calling(user_input, agent)
        
        # ==================================================================
        # Example 3: Question that doesn't require tools (Automatic Handling)
        print("\n" + "="*60)
        print("\nExample 3: No tool needed")
        print("-" * 40)
        
        user_input = "What's the capital of France?"
        execute_automatic_tool_calling(user_input, agent)
        
        # ==================================================================
        # Example 4: Complex multi-step reasoning
        print("\n" + "="*60)
        print("\nExample 4: Complex reasoning")
        print("-" * 40)
        
        user_input = "Compare the weather between Paris and Rome, and tell me which one would be better for outdoor activities today."
        execute_automatic_tool_calling(user_input, agent)
            
    except KeyError as e:
        print(f"Error: Missing environment variable {e}")
        print("Please ensure your .env file contains:")
        print("- AZURE_OPENAI_ENDPOINT")
        print("- AZURE_OPENAI_DEPLOYMENT_NAME") 
        print("- AZURE_OPENAI_API_VERSION")


if __name__ == "__main__":
    main()