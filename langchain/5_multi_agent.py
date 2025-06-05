import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored output
init(autoreset=True)

load_dotenv()

def get_ingredients() -> str:
    """Get a list of ingredients"""
    return f"Here are some ingredients you can use: macaroni, beef, cheese, broccoli, chicken, flour, cream cheese, mushrooms, anchovies."

model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

recipe_generator_assistant = create_react_agent(
    model=model,
    tools=[get_ingredients],
    prompt="You are a recipe generation assistant.  Consider the user's input when creating a recipe and also use ingredients that you have on hand.",
    name="recipe_generator_assistant"
)

gluten_free_reviewer_assistant = create_react_agent(
    model=model,
    tools=[],
    prompt="Your goal is to review a recipe and ensure it is gluten-free if the user indicates that they have a gluten sensitivity. " \
    "You DO NOT generate new recipes. " \
    "You ONLY suggest substitutions for ingredients that contain gluten." \
    "You ONLY concern yourself with gluten-containing ingredients, such as wheat, barley, rye, and products made from these grains. You DO NOT concern yourself with other dietary restrictions.",
    name="gluten_free_reviewer_assistant"
)

vegan_reviewer_assistant = create_react_agent(
    model=model,
    tools=[],
    prompt="Your goal is to review a recipe and ensure it is vegan if the user indicates that they have a preference for vegan cuisine. " \
    "You DO NOT generate new recipes. " \
    "You ONLY suggest substitutions for ingredients that contain non-vegan products." \
    "You ONLY concern yourself with non-vegan ingredients, such as meat, dairy, eggs, and honey. You DO NOT concern yourself with other dietary restrictions.",
    name="vegan_reviewer_assistant"
)

memory = MemorySaver()

supervisor = create_supervisor(
    agents=[recipe_generator_assistant, gluten_free_reviewer_assistant, vegan_reviewer_assistant],
    model=model,
    prompt=(
        "You manage a team of agents responsible for creating and reviewing a recipe.  You always need to ensure that the user's dietary restrictions are respected. If recipe substitutions have been made, " \
        "the recipe generator assistant needs to recreate the recipe with the substitutions. "
    )
).compile(checkpointer=memory)

# Global variable to track the last displayed message index for supervisor
last_supervisor_message_index = -1

def print_messages(messages, start_index=0, only_print_tool_calls=False):
    """
    Loop through an array of messages and print the content of any AIMessage objects.
    
    Args:
        messages: List of message objects that may contain AIMessage instances
        start_index: Index to start from (only print messages after this index)
    """
    for i, message in enumerate(messages[start_index:], start=start_index):
        if not only_print_tool_calls and isinstance(message, AIMessage):
            if message.content:
                print(f"{Fore.MAGENTA}     💬 Content: ", end="")
                print(message.content)
                print()
        elif isinstance(message, ToolMessage):
            print(f"{Fore.MAGENTA}     🔧 Tool Call: ", end="")
            print(message.content)
            print()

def print_agent_interaction(chunk, step_name):
    """Helper function to format and print agent interactions clearly"""
    global last_supervisor_message_index
    
    print(f"{Fore.CYAN}🔄 {step_name}:")
    
    # check if chunk contains a property named 'supervisor'
    if 'supervisor' in chunk:
        # print the supervisor's response
        print(f"{Fore.YELLOW}   📋 Supervisor is speaking")
        messages = chunk['supervisor']['messages']
        
        # Only print new messages since the last displayed index
        new_start_index = last_supervisor_message_index + 1
        if new_start_index < len(messages):
            print_messages(messages, start_index=new_start_index)
            # Update the last displayed message index
            last_supervisor_message_index = len(messages) - 1

    elif 'recipe_generator_assistant' in chunk:
        print(f"{Fore.GREEN}   🍳 Recipe Generator Assistant is thinking...")
        messages = chunk['recipe_generator_assistant']['messages']
        print_messages(messages, only_print_tool_calls=True)

    elif 'gluten_free_reviewer_assistant' in chunk:
        print(f"{Fore.BLUE}   🔍 Gluten Free Reviewer Assistant is thinking...")
        messages = chunk['gluten_free_reviewer_assistant']['messages']
        print_messages(messages, only_print_tool_calls=True)

    elif 'vegan_reviewer_assistant' in chunk:
        print(f"{Fore.BLUE}   🔍 Vegan Reviewer Assistant is thinking...")
        messages = chunk['vegan_reviewer_assistant']['messages']
        print_messages(messages, only_print_tool_calls=True)
    
    print(f"{Fore.WHITE}" + "-" * 70)

print("="*70)
print(f"{Fore.CYAN}{Style.BRIGHT}🍝 MULTI-AGENT RECIPE CONVERSATION CHAT")
print("="*70)
print(f"{Fore.GREEN}Start chatting with the supervisor agent!")
print(f"{Fore.YELLOW}Try: 'I have some macaroni, cheese and broccoli. Can you create a recipe for me?'")
print(f"{Fore.RED}Type 'exit' to end the conversation")
print("="*70)

thread_config = RunnableConfig(configurable={"thread_id": "1"})

while True:
    # Get user input
    user_input = input("\n👤 You: ").strip()
    
    # Check if user wants to exit
    if user_input.lower() == 'exit':
        print(f"\n{Fore.GREEN} Thanks for chatting! Goodbye!")
        break
    
    print(f"\n{Fore.CYAN} Processing your request...")
    print("-" * 70)
    
    # Stream the supervisor's response
    user_message = {
        "messages": [
            {
                "role": "user",
                "content": user_input
            }
        ]
    }
    
    # Stream the supervisor's response and process each step
    step_count = 0
    for chunk in supervisor.stream(user_message, config=thread_config):
        step_count += 1
        print_agent_interaction(chunk, f"Step {step_count}")

print(f"\n{Fore.CYAN}🏁 CHAT SESSION COMPLETE")
print("="*70)