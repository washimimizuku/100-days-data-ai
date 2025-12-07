# Day 83: LangChain Basics

## 📖 Learning Objectives (15 min)

**Time**: 1 hour

- Understand LangChain framework and its components
- Master chains for sequential LLM operations
- Learn prompt templates and variable substitution
- Implement memory management for conversations
- Build RAG systems with LangChain

---

## What is LangChain?

**LangChain** is a framework for developing applications powered by language models. It provides abstractions and tools to build complex LLM workflows.

**Core Components**: Models, Prompts, Chains, Memory, Agents, Tools

**Why LangChain**: Structured, reusable components instead of manual prompt management and parsing.

```bash
pip install langchain langchain-community ollama
```

## LLM Integration

```python
from langchain_community.llms import Ollama

llm = Ollama(model="mistral")
response = llm.invoke("What is machine learning?")

# Streaming
for chunk in llm.stream("Write a short story"):
    print(chunk, end="", flush=True)
```

---

## Prompt Templates

```python
from langchain.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate

# Basic Template
template = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms."
)
prompt = template.format(topic="quantum computing")

# Chat Template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("user", "Tell me about {topic}")
])

# Few-Shot Template
examples = [{"input": "happy", "output": "sad"}, {"input": "hot", "output": "cold"}]
few_shot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}"
    ),
    prefix="Give the opposite:",
    suffix="Input: {input}\nOutput:",
    input_variables=["input"]
)
```

---

## Chains

```python
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

# Simple Chain
chain = LLMChain(llm=llm, prompt=template)
result = chain.run(topic="neural networks")

# Sequential Chain (output of chain1 → input of chain2)
chain1 = LLMChain(llm=llm, prompt=PromptTemplate(
    input_variables=["subject"],
    template="Suggest a specific topic about {subject}"
))
chain2 = LLMChain(llm=llm, prompt=PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in detail"
))
overall_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)
result = overall_chain.run("machine learning")

# Sequential Chain with Multiple I/O
sentiment_chain = LLMChain(llm=llm, prompt=PromptTemplate(
    input_variables=["text"],
    template="Analyze sentiment: {text}\nSentiment:"
), output_key="sentiment")

response_chain = LLMChain(llm=llm, prompt=PromptTemplate(
    input_variables=["text", "sentiment"],
    template="Text: {text}\nSentiment: {sentiment}\nResponse:"
), output_key="response")

overall = SequentialChain(
    chains=[sentiment_chain, response_chain],
    input_variables=["text"],
    output_variables=["sentiment", "response"]
)
```

---

## Memory

```python
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain.chains import ConversationChain

# Buffer Memory - stores all messages
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
conversation.predict(input="Hi, I'm learning about AI")
conversation.predict(input="What did I just say?")  # Remembers

# Window Memory - keeps only last k messages
memory = ConversationBufferWindowMemory(k=2)
conversation = ConversationChain(llm=llm, memory=memory)

# Summary Memory - summarizes conversation to save tokens
memory = ConversationSummaryMemory(llm=llm)
conversation = ConversationChain(llm=llm, memory=memory)
```

---

## Document Loaders & Text Splitters

```python
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

# Load documents
loader = TextLoader("document.txt")
documents = loader.load()

loader = DirectoryLoader("./docs", glob="**/*.txt")
documents = loader.load()

# Split text
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n")
chunks = splitter.split_text(long_text)

# Recursive splitter (tries multiple separators)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ". ", " "]
)
chunks = splitter.split_text(long_text)
```

## Vector Stores

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="mistral")
vectorstore = Chroma.from_texts(
    texts=["ML is AI", "Python programming", "Data science"],
    embedding=embeddings
)
results = vectorstore.similarity_search("machine learning", k=2)
```

---

## RAG with LangChain

```python
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain, LLMChain

# Basic RAG
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever
)
result = qa_chain.run("What is machine learning?")

# RAG with Sources
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever
)
result = qa_chain({"question": "What is ML?"})
print(f"Answer: {result['answer']}\nSources: {result['sources']}")

# Custom RAG
rag_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Answer based on context:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
)
rag_chain = LLMChain(llm=llm, prompt=rag_template)

def rag_query(question):
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    return rag_chain.run(context=context, question=question)
```

---

## Output Parsers

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Define schema
response_schemas = [
    ResponseSchema(name="topic", description="The main topic"),
    ResponseSchema(name="summary", description="A brief summary")
]
parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Create prompt with format instructions
prompt = PromptTemplate(
    template="Analyze this text:\n{text}\n{format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(text="Machine learning is a subset of AI...")
parsed = parser.parse(result)  # {'topic': '...', 'summary': '...'}
```

---

## 💻 Exercises (40 min)

### Exercise 1: Prompt Templates
Create reusable prompt templates with variable substitution.

### Exercise 2: Sequential Chain
Build a multi-step chain for document analysis.

### Exercise 3: Conversation Memory
Implement a chatbot with conversation history.

### Exercise 4: RAG System
Build a complete RAG system with LangChain.

### Exercise 5: Output Parser
Create structured output parsing for data extraction.

---

## ✅ Quiz

Test your understanding in `quiz.md`

---

## 🎯 Key Takeaways

- LangChain provides abstractions for LLM applications
- Prompt templates enable reusable, parameterized prompts
- Chains connect multiple LLM calls sequentially
- Memory maintains conversation context
- Document loaders handle various file formats
- Text splitters chunk documents for RAG
- Vector stores enable semantic search
- RetrievalQA chains implement RAG patterns
- Output parsers structure LLM responses
- LangChain simplifies complex LLM workflows
- Components are modular and composable
- Supports multiple LLM providers (Ollama, OpenAI, etc.)

---

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain Cookbook](https://github.com/langchain-ai/langchain/tree/master/cookbook)
- [LangChain Templates](https://github.com/langchain-ai/langchain/tree/master/templates)
- [LangChain Community](https://github.com/langchain-ai/langchain)

---

## Tomorrow: Day 84 - Mini Project: RAG System

Build a complete production-ready RAG system integrating all concepts from Week 12: document processing, embeddings, vector databases, retrieval strategies, and LangChain.
