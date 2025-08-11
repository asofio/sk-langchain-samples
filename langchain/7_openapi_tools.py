import os
import json
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.agent_toolkits.openapi import planner
from langchain_community.utilities.requests import RequestsWrapper
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

# This example can be altered to load an OpenAPI spec directly from a URL as opposed to loading from a file.
# If you want to load from a URL, you can use the requests library to fetch the OpenAPI spec and then pass it to the `reduce_openapi_spec` function.
# For example:
#
# import requests
# response = requests.get("https://somewebaddress/openapi.json")
# api_spec = reduce_openapi_spec(response.json())

# The functionality below shows an example of loading an OpenAPI spec from a local file.
with open("[absolute-path-to-file]/7a_openapi_spec.json") as f:
    raw_api_spec = json.load(f)
    
api_spec = reduce_openapi_spec(raw_api_spec)

requests_wrapper = RequestsWrapper()

openapi_agent = planner.create_openapi_agent(
    api_spec,
    requests_wrapper,
    llm,
    allow_dangerous_requests=True,
)

user_query = (
    "Multiply 7 * 12 then add 3 and divide by 42."
)

openapi_agent.invoke(user_query)