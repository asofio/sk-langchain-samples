import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()

# Initialize the LLM with Azure
model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

# Simple prompt
#response = model.invoke("Tell me three facts about the Moon.")
for response in model.stream("Tell me three facts about Kangaroos."):
        print(response.content, end="")  # Print the response content as it streams in
