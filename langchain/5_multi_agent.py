import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

load_dotenv()

# def book_hotel(hotel_name: str):
#     """Book a hotel"""
#     return f"Successfully booked a stay at {hotel_name}."

# def book_flight(from_airport: str, to_airport: str):
#     """Book a flight"""
#     return f"Successfully booked a flight from {from_airport} to {to_airport}."

def get_ingredients() -> str:
    """Get a list of ingredients"""
    return f"Here are some ingredients you can use: macaroni, cheese, broccoli, chicken, cream cheese, mushrooms, anchovies."

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
    prompt="Your goal is to review a recipe and ensure it is gluten-free if the user indicates that they have a gluten sensitivity. You DO NOT generate new recipes. You ONLY suggest substitutions for ingredients that contain gluten.",
    name="gluten_free_reviewer_assistant"
)

thread_config = {"configurable": {"thread_id": "recipe_conversation_1"}}

memory = MemorySaver()

supervisor = create_supervisor(
    agents=[recipe_generator_assistant, gluten_free_reviewer_assistant],
    model=model,
    prompt=(
        "You manage a team of agents responsible for creating and reviewing a recipe.  You always need to ensure that the user's dietary restrictions are respected. If recipe substitutions have been made, " \
        "the recipe generator assistant needs to recreate the recipe with the substitutions. " \
    )
).compile(checkpointer=memory)
config = RunnableConfig(configurable={"thread_id": "1"})

# Global variable to track the last displayed message index for supervisor
last_supervisor_message_index = -1

def print_ai_messages(messages, start_index=0):
    """
    Loop through an array of messages and print the content of any AIMessage objects.
    
    Args:
        messages: List of message objects that may contain AIMessage instances
        start_index: Index to start from (only print messages after this index)
    """
    for i, message in enumerate(messages[start_index:], start=start_index):
        if isinstance(message, AIMessage):
            print(f"     Content: {message.content}")
            print()

def print_agent_interaction(chunk, step_name):
    """Helper function to format and print agent interactions clearly"""
    global last_supervisor_message_index
    
    print(f"🔄 {step_name}:")
    
    # check if chunk contains a property named 'supervisor'
    if 'supervisor' in chunk:
        # print the supervisor's response
        print(f"   Supervisor is speaking")
        messages = chunk['supervisor']['messages']
        
        # Only print new messages since the last displayed index
        new_start_index = last_supervisor_message_index + 1
        if new_start_index < len(messages):
            print_ai_messages(messages, start_index=new_start_index)
            # Update the last displayed message index
            last_supervisor_message_index = len(messages) - 1
    elif 'recipe_generator_assistant' in chunk:
        print(f"   Recipe Generator Assistant is thinking...")
    elif 'gluten_free_reviewer_assistant' in chunk:
        print(f"   Gluten Free Reviewer Assistant is thinking...")
    
    print("-" * 70)

print("="*70)
print("🍝 MULTI-AGENT RECIPE CONVERSATION CHAT")
print("="*70)
print("💬 Start chatting with the supervisor agent!")
print("💡 Try: 'I have some macaroni, cheese and broccoli. Can you create a recipe for me?'")
print("🚪 Type 'exit' to end the conversation")
print("="*70)

while True:
    # Get user input
    user_input = input("\n👤 You: ").strip()
    
    # Check if user wants to exit
    if user_input.lower() == 'exit':
        print("\n👋 Thanks for chatting! Goodbye!")
        break
    
    # Skip empty inputs
    if not user_input:
        continue
    
    print(f"\n🤖 Processing your request...")
    print("-" * 70)
    
    # Stream the supervisor's response
    step_count = 0
    for chunk in supervisor.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        },
        config=config
    ):
        step_count += 1
        print_agent_interaction(chunk, f"Step {step_count}")

print("\n🏁 CHAT SESSION COMPLETE")
print("="*70)