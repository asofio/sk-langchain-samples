import os
import asyncio
from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.memory import VolatileMemoryStore

load_dotenv()

async def create_embeddings_demo():
    """
    Simple embeddings demo that shows how to retrieve relevant documents 
    based on semantic similarity using embeddings.
    """
    
    # Initialize Azure OpenAI embeddings
    embeddings = AzureTextEmbedding(
        deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )
    
    # Create an in-memory vector store
    memory_store = VolatileMemoryStore()
    
    # Create semantic memory using the embeddings and memory store
    semantic_memory = SemanticTextMemory(memory_store, embeddings)
    
    # Sample documents that contain information about different concepts
    # These simulate a small database of knowledge
    documents = [
        ("doc1", "A prince is a male royal family member, typically the son of a king or queen. Princes often inherit titles and may rule kingdoms."),
        ("doc2", "A princess is a female royal family member, typically the daughter of a king or queen. Princesses hold noble titles in monarchies."),
        ("doc3", "A king is the male ruler of a kingdom, holding supreme authority over the land and its people. Kings wear crowns as symbols of power."),
        ("doc4", "A queen is the female ruler of a kingdom, or the wife of a king. Queens may rule independently or alongside their royal spouse."),
        ("doc5", "Water is a transparent, colorless liquid essential for all life. It covers most of Earth's surface and exists in oceans, rivers, and lakes."),
        ("doc6", "A boy is a young male human, typically under the age of adulthood. Boys grow and develop into men as they mature."),
        ("doc7", "A man is an adult male human being. Men play various roles in society as fathers, workers, leaders, and community members."),
        ("doc8", "A woman is an adult female human being. Women contribute to society in countless ways as mothers, professionals, leaders, and innovators."),
        ("doc9", "A girl is a young female human, typically under the age of adulthood. Girls develop and grow into women as they mature.")
    ]
    
    # Add documents to semantic memory
    collection_name = "knowledge_base"
    for doc_id, content in documents:
        await semantic_memory.save_information(
            collection=collection_name,
            text=content,
            id=doc_id
        )
    
    # Demo: Query for "What is a Prince?" and show similar documents
    query = "What is a Prince?"
    print(f"🔍 Query: {query}")
    print("=" * 50)
    
    # Retrieve similar documents
    search_results = await semantic_memory.search(
        collection=collection_name,
        query=query,
        limit=4,
        min_relevance_score=0.0
    )
    
    print("📄 Most relevant documents:")
    for i, result in enumerate(search_results, 1):
        text_preview = result.text[:100] if result.text else "No text available"
        print(f"\n{i}. {text_preview}...")
    
    # Show similarity scores for more detailed analysis
    print("\n" + "=" * 50)
    print("📊 Similarity scores:")
    for result in search_results:
        text_preview = result.text[:60] if result.text else "No text available"
        print(f"Score: {result.relevance:.4f} | {text_preview}...")
    
    # Demo: Show how embeddings capture semantic relationships
    print("\n" + "=" * 50)
    print("🧠 Semantic relationship demo:")
    
    # Compare different queries
    queries = [
        "royal family member",
        "young person", 
        "liquid substance",
        "male ruler"
    ]
    
    for test_query in queries:
        print(f"\nQuery: '{test_query}'")
        top_results = await semantic_memory.search(
            collection=collection_name,
            query=test_query,
            limit=1
        )
        if top_results:
            top_result = top_results[0]
            text_preview = top_result.text[:80] if top_result.text else "No text available"
            print(f"Top match: {text_preview}...")
            print(f"Relevance: {top_result.relevance:.4f}")

async def main():
    print("🌟 Simple Embeddings Demo with Semantic Kernel")
    print("This demo shows how embeddings can retrieve semantically similar documents")
    print()
    
    try:
        await create_embeddings_demo()
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have set the following environment variables:")
        print("- AZURE_OPENAI_ENDPOINT")
        print("- AZURE_OPENAI_API_VERSION")
        print("- AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

if __name__ == "__main__":
    asyncio.run(main())
