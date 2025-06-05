import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from colorama import init, Fore

init(autoreset=True)
load_dotenv()

model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

# Example 1: Simple Chain
print("=" * 50)
print(f"{Fore.GREEN}Example 1: Simple Chain")

joke_prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")

chain = joke_prompt | model | StrOutputParser()

response = chain.invoke({"topic": "bears"})

print(f"Response: {response}")
print("=" * 50)

# ========================================

# Example 2: More Complex Chain with Intermediate Steps
print(f"{Fore.GREEN}Example 2: More Complex Chain with Intermediate Steps")

story_prompt = ChatPromptTemplate.from_template("Tell me a short story about {topic}")
joke_prompt = ChatPromptTemplate.from_template("Tell me a joke about this story: {story}")

more_complex_chain = story_prompt | model | StrOutputParser() | RunnableLambda(lambda x: {"story": x}) | joke_prompt | model | StrOutputParser()

response = more_complex_chain.invoke({"topic": "bears"})

print(f"Response: {response}")
print("=" * 50)

# =========================================

# Example 3: Using RunnablePassthrough for Intermediate Steps to capture results
print(f"{Fore.GREEN}Example 3: Using RunnablePassthrough for Intermediate Steps to capture results")

chain = (
    RunnablePassthrough.assign(
        story=(story_prompt | model | StrOutputParser())  # save story as an intermediate key
    )
    .assign(
        joke=(lambda x: (joke_prompt | model | StrOutputParser()).invoke({"story": x["story"]}))  # use the story to generate joke
    )
)

response = chain.invoke({"topic": "bears"})

print(f"Story: {response['story']}")
print("=" * 10)
print(f"Joke: {response['joke']}")
print("=" * 50)

# =========================================

# Example 4: Streaming Responses
print(f"{Fore.GREEN}Example 4: Streaming Responses")

# Stream story generation
print(f"{Fore.BLUE}Story: ", end="", flush=True)

story_chain = story_prompt | model | StrOutputParser()
story_text = ""

for chunk in story_chain.stream({"topic": "bears"}):
    print(chunk, end="", flush=True)
    story_text += chunk

print(f"\n\n{Fore.BLUE}Joke: ", end="", flush=True)

# Stream joke generation
joke_chain = joke_prompt | model | StrOutputParser()

for chunk in joke_chain.stream({"story": story_text}):
    print(chunk, end="", flush=True)