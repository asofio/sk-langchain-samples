import os
import tempfile
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

class CookingAssistantRAG:
    """A RAG-based cooking assistant that can answer questions about recipes and cooking techniques."""
    
    def __init__(self):
        """Initialize the cooking assistant with Azure OpenAI models and sample recipe data."""
        try:
            # Initialize Azure OpenAI models
            self.llm = AzureChatOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
                api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                temperature=0.3,  # Slightly creative but consistent
            )
            
            # Initialize embeddings model
            # Note: You may need to configure a separate embedding deployment
            self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
                api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            )
            
            # Sample cooking knowledge base
            self.recipe_data = self._get_sample_recipes()
            
            # Initialize the vector store
            self.vectorstore = self._create_vectorstore()
            
            # Create the RAG chain
            self.rag_chain = self._create_rag_chain()
            
        except KeyError as e:
            raise ValueError(f"Missing required environment variable: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CookingAssistantRAG: {e}")
    
    def _get_sample_recipes(self) -> list:
        """Get sample recipe data for the knowledge base."""
        recipes = [
            {
                "title": "Classic Chocolate Chip Cookies",
                "content": """
                Ingredients:
                - 2 1/4 cups all-purpose flour
                - 1 tsp baking soda
                - 1 tsp salt
                - 1 cup butter, softened
                - 3/4 cup granulated sugar
                - 3/4 cup brown sugar
                - 2 large eggs
                - 2 tsp vanilla extract
                - 2 cups chocolate chips
                
                Instructions:
                1. Preheat oven to 375°F (190°C)
                2. Mix flour, baking soda, and salt in a bowl
                3. Cream butter and sugars until fluffy
                4. Beat in eggs and vanilla
                5. Gradually mix in flour mixture
                6. Stir in chocolate chips
                7. Drop rounded tablespoons onto ungreased baking sheets
                8. Bake 9-11 minutes until golden brown
                9. Cool on baking sheet for 2 minutes, then transfer to wire rack
                
                Tips: Don't overbake! Cookies continue cooking on the hot pan.
                """
            },
            {
                "title": "Perfect Pasta Carbonara",
                "content": """
                Ingredients:
                - 400g spaghetti
                - 200g pancetta or guanciale, diced
                - 4 large eggs
                - 100g Pecorino Romano cheese, grated
                - Black pepper, freshly ground
                - Salt for pasta water
                
                Instructions:
                1. Bring large pot of salted water to boil
                2. Cook pasta until al dente (1-2 minutes less than package directions)
                3. While pasta cooks, crisp pancetta in large skillet
                4. Whisk eggs, cheese, and pepper in large bowl
                5. Reserve 1 cup pasta water before draining
                6. Add hot pasta to pancetta pan
                7. Remove from heat, add egg mixture while tossing
                8. Add pasta water gradually until creamy
                9. Serve immediately with extra cheese and pepper
                
                Key tip: The heat from the pasta cooks the eggs. Too much heat will scramble them!
                """
            },
            {
                "title": "Homemade Pizza Dough",
                "content": """
                Ingredients:
                - 3 cups bread flour
                - 1 tsp instant yeast
                - 1 1/4 tsp salt
                - 1 tbsp olive oil
                - 1 cup warm water
                
                Instructions:
                1. Mix flour, yeast, and salt in large bowl
                2. Add water and oil, mix until shaggy dough forms
                3. Knead on floured surface for 8-10 minutes until smooth
                4. Place in oiled bowl, cover, rise 1-2 hours until doubled
                5. Punch down, divide into 2 portions for thin crust or keep whole for thick
                6. Let rest 15 minutes before rolling
                7. Roll out and add toppings
                8. Bake at 475°F (245°C) for 10-15 minutes
                
                Pro tips: High heat is key! Use a pizza stone if you have one.
                """
            },
            {
                "title": "Basic Knife Skills",
                "content": """
                Essential knife techniques for cooking:
                
                Knife Types:
                - Chef's knife: 8-10 inch blade, most versatile
                - Paring knife: 3-4 inch blade, for small tasks
                - Serrated knife: For bread and tomatoes
                
                Basic Cuts:
                - Julienne: Thin matchstick cuts (1/8 inch thick)
                - Dice: Uniform cubes (small 1/4 inch, medium 1/2 inch, large 3/4 inch)
                - Chiffonade: Thin ribbon cuts for herbs and leafy greens
                - Brunoise: Very fine dice (1/8 inch cubes)
                
                Safety Tips:
                - Keep knives sharp (dull knives are more dangerous)
                - Use proper cutting board
                - Keep fingers curled when cutting
                - Cut away from your body
                - Clean knives immediately after use
                
                The rocking motion with a chef's knife is most efficient for chopping.
                """
            },
            {
                "title": "Roasted Vegetables Guide",
                "content": """
                Perfect roasted vegetables every time:
                
                Temperature: 425°F (220°C) for most vegetables
                
                Timing by vegetable:
                - Root vegetables (carrots, potatoes): 25-35 minutes
                - Brussels sprouts, cauliflower: 20-25 minutes
                - Broccoli, asparagus: 12-15 minutes
                - Bell peppers, zucchini: 15-20 minutes
                - Onions: 20-30 minutes
                
                Preparation:
                1. Cut vegetables into uniform pieces
                2. Toss with olive oil, salt, and pepper
                3. Spread in single layer on baking sheet
                4. Don't overcrowd - use multiple pans if needed
                5. Flip halfway through cooking
                
                Seasoning ideas:
                - Mediterranean: Rosemary, thyme, garlic
                - Asian: Soy sauce, sesame oil, ginger
                - Mexican: Cumin, chili powder, lime
                
                Caramelization is key - don't stir too often!
                """
            }
        ]
        return recipes
    
    def _create_vectorstore(self):
        """Create and populate the vector store with recipe documents."""
        try:
            # Convert recipes to documents
            documents = []
            for recipe in self.recipe_data:
                doc = Document(
                    page_content=f"Recipe: {recipe['title']}\n\n{recipe['content']}",
                    metadata={"title": recipe["title"], "type": "recipe"}
                )
                documents.append(doc)
            
            # Split documents into chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            vectorstore = InMemoryVectorStore.from_documents(splits, self.embeddings)
            
            print(f"Created vector store with {len(splits)} document chunks")
            return vectorstore
            
        except Exception as e:
            raise RuntimeError(f"Failed to create vector store: {e}")
    
    def _create_rag_chain(self):
        """Create the RAG chain with prompt template."""
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Retrieve top 3 most similar chunks
        )
        
        # Create prompt template
        template = """You are an expert cooking assistant. Use the following recipe and cooking information to answer the user's question. 
        If the information isn't in the provided context, say so and provide general cooking advice if appropriate.

        Context:
        {context}

        Question: {question}

        Answer: Provide a helpful, detailed response based on the context above. Include specific steps, measurements, or techniques when relevant."""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the RAG chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def ask_question(self, question: str) -> str:
        """Ask a cooking-related question and get a RAG-enhanced response."""
        try:
            print(f"🍳 Question: {question}")
            print("🔍 Searching recipe database...")
            
            # Get relevant documents for context
            docs = self.vectorstore.similarity_search(question, k=3)
            print(f"📚 Found {len(docs)} relevant recipe chunks")
            
            # Generate response using RAG chain
            response = self.rag_chain.invoke(question)
            
            print(f"🤖 Answer: {response}")
            return response
            
        except Exception as e:
            error_msg = f"Error processing question: {e}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def show_available_recipes(self):
        """Display all available recipes in the knowledge base."""
        print("📖 Available recipes and guides:")
        for i, recipe in enumerate(self.recipe_data, 1):
            print(f"{i}. {recipe['title']}")

def main():
    """Main function demonstrating the RAG cooking assistant."""
    try:
        print("🍳 Initializing Cooking Assistant RAG System...")
        assistant = CookingAssistantRAG()
        print("✅ Assistant ready!")
        
        # Show available recipes
        assistant.show_available_recipes()
        
        print("\n" + "="*60)
        print("RAG COOKING ASSISTANT DEMO")
        print("="*60)
        
        # Example questions
        questions = [
            "How do I make chocolate chip cookies?",
            "What's the secret to perfect carbonara?",
            "How long should I roast broccoli?",
            "What knife should I use for chopping vegetables?",
            "How do I prevent my pasta from getting mushy?",
            "What temperature is best for making pizza?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n--- Example {i} ---")
            assistant.ask_question(question)
            
            if i < len(questions):
                input("\nPress Enter to continue to next example...")
        
        print("\n" + "="*60)
        print("✨ Interactive Mode")
        print("="*60)
        print("Ask me any cooking question! (type 'quit' to exit)")
        
        while True:
            user_question = input("\n🍳 Your question: ").strip()
            if user_question.lower() in ['quit', 'exit', 'q']:
                print("👋 Happy cooking!")
                break
            elif user_question:
                assistant.ask_question(user_question)
            
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("Please ensure your .env file contains:")
        print("- AZURE_OPENAI_ENDPOINT")
        print("- AZURE_OPENAI_DEPLOYMENT_NAME") 
        print("- AZURE_OPENAI_API_VERSION")
        print("- AZURE_OPENAI_EMBEDDING_DEPLOYMENT (optional, defaults to text-embedding-ada-002)")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
