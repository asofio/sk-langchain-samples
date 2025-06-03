import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

load_dotenv()

def create_embeddings_demo():
    """
    Simple embeddings demo that shows how to retrieve relevant documents 
    based on semantic similarity using embeddings.
    """
    
    # Initialize Azure OpenAI embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )
    
    # Sample documents that contain information about different concepts
    # These simulate a small database of knowledge
    documents = [
        Document(page_content="A prince is a male royal family member, typically the son of a king or queen. Princes often inherit titles and may rule kingdoms."),
        Document(page_content="A princess is a female royal family member, typically the daughter of a king or queen. Princesses hold noble titles in monarchies."),
        Document(page_content="A king is the male ruler of a kingdom, holding supreme authority over the land and its people. Kings wear crowns as symbols of power."),
        Document(page_content="A queen is the female ruler of a kingdom, or the wife of a king. Queens may rule independently or alongside their royal spouse."),
        Document(page_content="Water is a transparent, colorless liquid essential for all life. It covers most of Earth's surface and exists in oceans, rivers, and lakes."),
        Document(page_content="A boy is a young male human, typically under the age of adulthood. Boys grow and develop into men as they mature."),
        Document(page_content="A man is an adult male human being. Men play various roles in society as fathers, workers, leaders, and community members."),
        Document(page_content="A woman is an adult female human being. Women contribute to society in countless ways as mothers, professionals, leaders, and innovators."),
        Document(page_content="A girl is a young female human, typically under the age of adulthood. Girls develop and grow into women as they mature."),
    ]
    
    # Create vector store and add documents
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents)
    
    # Demo: Query for "What is a Prince?" and show similar documents
    query = "What is a Prince?"
    print(f"🔍 Query: {query}")
    print("=" * 50)
    
    # Retrieve similar documents
    similar_docs = vector_store.similarity_search(query, k=4)
    
    print("📄 Most relevant documents:")
    for i, doc in enumerate(similar_docs, 1):
        print(f"\n{i}. {doc.page_content[:100]}...")
    
    # Show similarity scores for more detailed analysis
    print("\n" + "=" * 50)
    print("📊 Similarity scores:")
    
    docs_with_scores = vector_store.similarity_search_with_score(query, k=4)
    for doc, score in docs_with_scores:
        print(f"Score: {score:.4f} | {doc.page_content[:60]}...")
    
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
        score_with_top = vector_store.similarity_search_with_score(test_query, k=1)[0]
        print(f"Top match: {score_with_top[0].page_content[:80]}...")
        print(f"Relevance score: {score_with_top[1]:.4f}")

if __name__ == "__main__":
    print("🌟 Simple Embeddings Demo with LangChain")
    print("This demo shows how embeddings can retrieve semantically similar documents")
    print()
    
    try:
        create_embeddings_demo()
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have set the following environment variables:")
        print("- AZURE_OPENAI_ENDPOINT")
        print("- AZURE_OPENAI_API_VERSION")
        print("- AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
