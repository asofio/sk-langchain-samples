import os
import random
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

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
    """Main function demonstrating automatic tool calling with Azure OpenAI."""
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
        
        # Create a prompt template for the agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that can check weather information. "
                      "Use the available tools to answer user questions about weather."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Create the tool-calling agent
        agent = create_tool_calling_agent(model, tools, prompt)
        
        # Create an agent executor that will automatically handle tool calls
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,  # Set to True to see the agent's reasoning
            handle_parsing_errors=True
        )
        
        print("="*50)
        print("Example 1: Single city weather")
        print("="*50)
        
        # Example 1: Single city
        user_input = "What's the weather like in Sydney, Australia?"
        print(f"User: {user_input}")
        
        # The agent executor will automatically:
        # 1. Determine if tools are needed
        # 2. Call the appropriate tools
        # 3. Process the results
        # 4. Generate a final response
        result = agent_executor.invoke({"input": user_input})
        print(f"\nFinal Answer: {result['output']}")
            
        print("\n" + "="*50)
        print("Example 2: Multiple cities")
        print("="*50)
        
        # Example 2: Multiple cities
        user_input = "Can you check the weather in Tokyo and London?"
        print(f"User: {user_input}")
        
        result = agent_executor.invoke({"input": user_input})
        print(f"\nFinal Answer: {result['output']}")
        
        print("\n" + "="*50)
        print("Example 3: No tool needed")
        print("="*50)
        
        # Example 3: Question that doesn't require tools
        user_input = "What's the capital of France?"
        print(f"User: {user_input}")
        
        result = agent_executor.invoke({"input": user_input})
        print(f"\nFinal Answer: {result['output']}")
            
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
