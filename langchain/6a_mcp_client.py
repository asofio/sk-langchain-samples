import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()

async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                # Replace with absolute path to your weather_server.py file
                "args": ["[your-absolute-path-here]/sk-langchain-samples/langchain/6c_mcp_server_weather.py"],
                "transport": "stdio",
            },
            "weather": {
                # Ensure you start your math server on port 8000
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        }
    )

    model = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )
    
    try:
        # Get tools from the MCP servers
        tools = await client.get_tools()
        
        # Create the agent with the tools
        agent = create_react_agent(
            model,
            tools
        )
        
        # Test math functionality
        print("Testing math functionality...")
        math_response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
        )
        print(f"Math response: {math_response}")
        
        # Test weather functionality
        print("\nTesting weather functionality...")
        weather_response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
        )
        print(f"Weather response: {weather_response}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())