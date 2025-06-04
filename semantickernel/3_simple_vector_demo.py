import os
import asyncio
from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding
from scipy.spatial.distance import cosine

load_dotenv()

async def generate_vector_embedding_demo():

    embeddings = AzureTextEmbedding(
        deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )
    
    word = "Prince"
    
    print(f"Generating embedding for the word: '{word}'")
    print()
    
    embedding_result = await embeddings.generate_embeddings([word])
    
    embedding_vector = embedding_result[0]
    
    print("Vector:", embedding_vector)
    print("Dimensionality of the embedding vector:", len(embedding_vector))
    print()

    other_words = ["Prince", "King", "Princess", "Royal", "Crown", "Peasant", "Pauper", "Algorithm", "Entropy", "Asphalt"]
    for related_word in other_words:
        related_embedding_result = await embeddings.generate_embeddings([related_word])
        if len(related_embedding_result) > 0:
            cosine_similarity = 1 - cosine(embedding_vector, related_embedding_result[0])
            print(f"   '{word}' vs '{related_word}': {cosine_similarity:.6f}")

async def main():
    try:
        await generate_vector_embedding_demo()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have set the following environment variables:")
        print("- AZURE_OPENAI_ENDPOINT")
        print("- AZURE_OPENAI_API_VERSION")
        print("- AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

if __name__ == "__main__":
    asyncio.run(main())
