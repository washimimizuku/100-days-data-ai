"""
Day 83: LangChain Basics - Solutions
"""

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


# Initialize LLM
llm = Ollama(model="mistral", temperature=0.7)


# Exercise 1: Prompt Templates
def exercise_1_prompt_templates():
    """Exercise 1: Prompt Templates"""
    print("Exercise 1: Prompt Templates")
    print("-" * 40)
    
    # Basic template
    basic_template = PromptTemplate(
        input_variables=["topic"],
        template="Explain {topic} in simple terms."
    )
    
    prompt = basic_template.format(topic="machine learning")
    print(f"\nBasic template:\n{prompt}")
    
    # Chat template
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI tutor."),
        ("user", "Teach me about {subject}")
    ])
    
    messages = chat_template.format_messages(subject="neural networks")
    print(f"\nChat template: {len(messages)} messages")
    
    # Few-shot template
    examples = [
        {"input": "ML", "output": "Machine Learning"},
        {"input": "AI", "output": "Artificial Intelligence"}
    ]
    
    example_template = PromptTemplate(
        input_variables=["input", "output"],
        template="Q: {input}\nA: {output}"
    )
    
    few_shot = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_template,
        prefix="Expand these acronyms:",
        suffix="Q: {input}\nA:",
        input_variables=["input"]
    )
    
    prompt = few_shot.format(input="DL")
    print(f"\nFew-shot template:\n{prompt[:100]}...")


# Exercise 2: Sequential Chain
def exercise_2_sequential_chain():
    """Exercise 2: Sequential Chain"""
    print("\nExercise 2: Sequential Chain")
    print("-" * 40)
    
    # Chain 1: Extract topic
    topic_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["text"],
            template="Extract the main topic from this text in 2-3 words:\n{text}\n\nTopic:"
        )
    )
    
    # Chain 2: Summarize
    summary_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["topic"],
            template="Write a 2-sentence summary about {topic}:"
        )
    )
    
    # Combine
    overall_chain = SimpleSequentialChain(
        chains=[topic_chain, summary_chain],
        verbose=False
    )
    
    # Run
    text = "Machine learning algorithms learn patterns from data to make predictions."
    print(f"\nInput: {text}")
    result = overall_chain.run(text)
    print(f"\nOutput: {result}")


# Exercise 3: Conversation Memory
class ConversationalBot:
    """Chatbot with memory."""
    
    def __init__(self):
        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(
            llm=llm,
            memory=self.memory,
            verbose=False
        )
    
    def chat(self, message: str) -> str:
        """Send message and get response."""
        return self.conversation.predict(input=message)
    
    def get_history(self) -> str:
        """Get conversation history."""
        return self.memory.buffer
    
    def clear(self):
        """Clear memory."""
        self.memory.clear()


def exercise_3_conversation_memory():
    """Exercise 3: Conversation Memory"""
    print("\nExercise 3: Conversation Memory")
    print("-" * 40)
    
    bot = ConversationalBot()
    
    # Conversation
    messages = [
        "Hi, I'm learning about RAG systems",
        "What did I just say I'm learning about?"
    ]
    
    for msg in messages:
        print(f"\nUser: {msg}")
        response = bot.chat(msg)
        print(f"Bot: {response[:100]}...")
    
    print(f"\nConversation history length: {len(bot.get_history())} chars")


# Exercise 4: RAG System
class SimpleRAG:
    """Simple RAG system with LangChain."""
    
    def __init__(self, documents: list):
        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        self.chunks = splitter.create_documents(documents)
        
        # Create embeddings
        embeddings = OllamaEmbeddings(model="mistral")
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 2}
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )
    
    def query(self, question: str) -> dict:
        """Query the RAG system."""
        result = self.qa_chain({"query": question})
        return {
            'answer': result['result'],
            'sources': [doc.page_content[:100] for doc in result['source_documents']]
        }


def exercise_4_rag_system():
    """Exercise 4: RAG System"""
    print("\nExercise 4: RAG System")
    print("-" * 40)
    
    # Sample documents
    documents = [
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers for complex tasks.",
        "RAG systems combine retrieval with generation for accurate responses.",
        "Vector databases store embeddings for fast similarity search."
    ]
    
    print(f"\nIndexing {len(documents)} documents...")
    rag = SimpleRAG(documents)
    
    # Query
    question = "What is RAG?"
    print(f"\nQuestion: {question}")
    
    result = rag.query(question)
    print(f"\nAnswer: {result['answer'][:150]}...")
    print(f"\nSources: {len(result['sources'])} documents")


# Exercise 5: Output Parser
def exercise_5_output_parser():
    """Exercise 5: Output Parser"""
    print("\nExercise 5: Output Parser")
    print("-" * 40)
    
    # Define schema
    response_schemas = [
        ResponseSchema(name="topic", description="The main topic"),
        ResponseSchema(name="summary", description="A brief summary"),
        ResponseSchema(name="keywords", description="Key terms (comma-separated)")
    ]
    
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = parser.get_format_instructions()
    
    # Create prompt
    prompt = PromptTemplate(
        template="Analyze this text:\n{text}\n\n{format_instructions}",
        input_variables=["text"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    # Chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run
    text = "Machine learning algorithms learn patterns from data to make predictions."
    print(f"\nInput: {text}")
    
    try:
        result = chain.run(text=text)
        print(f"\nRaw output:\n{result[:200]}...")
        
        # Parse
        parsed = parser.parse(result)
        print(f"\nParsed output:")
        for key, value in parsed.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"\nParsing note: {e}")
        print("(Output parsing may vary with LLM responses)")


if __name__ == "__main__":
    print("Day 83: LangChain Basics - Solutions\n")
    print("=" * 60)
    
    try:
        exercise_1_prompt_templates()
        exercise_2_sequential_chain()
        exercise_3_conversation_memory()
        exercise_4_rag_system()
        exercise_5_output_parser()
        
        print("\n" + "=" * 60)
        print("All exercises completed!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have installed:")
        print("pip install langchain langchain-community ollama chromadb")
        print("\nAnd Ollama is running with mistral model:")
        print("ollama pull mistral")
