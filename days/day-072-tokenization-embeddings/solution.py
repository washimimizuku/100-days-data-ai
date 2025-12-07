"""
Day 72: Tokenization & Embeddings - Solutions
"""

from transformers import GPT2Tokenizer, BertTokenizer, T5Tokenizer
from transformers import BertModel
import torch
import torch.nn.functional as F


def exercise_1_compare_tokenizers():
    """Exercise 1: Compare Tokenizers"""
    
    text = "Tokenization is fundamental for NLP"
    
    # Load tokenizers
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
    
    # Tokenize with each
    gpt2_tokens = gpt2_tokenizer.tokenize(text)
    bert_tokens = bert_tokenizer.tokenize(text)
    t5_tokens = t5_tokenizer.tokenize(text)
    
    # Print results
    print(f"Original text: '{text}'\n")
    
    print("GPT-2 (BPE):")
    print(f"  Tokens: {gpt2_tokens}")
    print(f"  Count: {len(gpt2_tokens)}")
    print()
    
    print("BERT (WordPiece):")
    print(f"  Tokens: {bert_tokens}")
    print(f"  Count: {len(bert_tokens)}")
    print()
    
    print("T5 (SentencePiece):")
    print(f"  Tokens: {t5_tokens}")
    print(f"  Count: {len(t5_tokens)}")
    print()
    
    # Encode to IDs
    print("Token IDs:")
    print(f"  GPT-2: {gpt2_tokenizer.encode(text)}")
    print(f"  BERT: {bert_tokenizer.encode(text)}")
    print(f"  T5: {t5_tokenizer.encode(text)}")


def exercise_2_subword_analysis():
    """Exercise 2: Subword Analysis"""
    
    words = ["unhappiness", "antidisestablishmentarianism", "COVID-19", 
             "preprocessing", "transformer"]
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    print("Subword Tokenization Analysis:\n")
    
    for word in words:
        tokens = tokenizer.tokenize(word)
        token_count = len(tokens)
        
        print(f"Word: '{word}'")
        print(f"  Tokens: {tokens}")
        print(f"  Count: {token_count}")
        print(f"  Avg chars/token: {len(word)/token_count:.1f}")
        print()


def exercise_3_embedding_similarity():
    """Exercise 3: Embedding Similarity"""
    
    sentences = [
        "The cat sits on the mat",
        "A feline rests on the rug",
        "Python is a programming language"
    ]
    
    # Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    
    def get_sentence_embedding(text):
        """Get sentence embedding using mean pooling"""
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling over sequence length
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings[0]
    
    def cosine_similarity(a, b):
        """Compute cosine similarity"""
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    
    # Get embeddings
    embeddings = [get_sentence_embedding(sent) for sent in sentences]
    
    print("Sentence Embeddings:\n")
    for i, sent in enumerate(sentences):
        print(f"{i+1}. {sent}")
        print(f"   Embedding shape: {embeddings[i].shape}")
    print()
    
    # Compute pairwise similarities
    print("Pairwise Similarities:\n")
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"Sentence {i+1} vs {j+1}: {sim:.4f}")
    
    # Find most similar pair
    max_sim = -1
    max_pair = (0, 0)
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > max_sim:
                max_sim = sim
                max_pair = (i, j)
    
    print(f"\nMost similar pair: Sentence {max_pair[0]+1} and {max_pair[1]+1}")
    print(f"Similarity: {max_sim:.4f}")


def exercise_4_vocabulary_analysis():
    """Exercise 4: Vocabulary Analysis"""
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    print("GPT-2 Tokenizer Vocabulary Analysis:\n")
    
    # Vocabulary size
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size:,}")
    print()
    
    # Look up specific words
    words = ["hello", "world", "tokenization", "AI", "😀"]
    print("Token IDs for common words:")
    for word in words:
        token_id = tokenizer.encode(word, add_special_tokens=False)
        tokens = tokenizer.tokenize(word)
        print(f"  '{word}' -> {tokens} -> {token_id}")
    print()
    
    # Special tokens
    print("Special tokens:")
    print(f"  BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  UNK token: {tokenizer.unk_token}")
    print()
    
    # Check if tokens exist
    test_tokens = ["Ġhello", "Ġworld", "ization"]
    print("Token existence check:")
    for token in test_tokens:
        exists = token in tokenizer.get_vocab()
        print(f"  '{token}': {exists}")


def exercise_5_token_statistics():
    """Exercise 5: Token Statistics"""
    
    paragraph = """
    Large language models have revolutionized natural language processing.
    These models use transformer architectures and are trained on massive
    amounts of text data. Tokenization is a crucial preprocessing step that
    converts text into numerical representations that models can process.
    """
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Tokenize
    tokens = tokenizer.tokenize(paragraph)
    token_ids = tokenizer.encode(paragraph)
    
    # Word count (approximate)
    words = paragraph.split()
    word_count = len([w for w in words if w.strip()])
    
    print("Token Statistics:\n")
    print(f"Original text length: {len(paragraph)} characters")
    print(f"Word count: {word_count}")
    print(f"Token count: {len(tokens)}")
    print(f"Token ID count: {len(token_ids)}")
    print(f"Average tokens per word: {len(tokens)/word_count:.2f}")
    print()
    
    # Token lengths
    token_lengths = [len(token) for token in tokens]
    print(f"Shortest token: '{min(tokens, key=len)}' ({min(token_lengths)} chars)")
    print(f"Longest token: '{max(tokens, key=len)}' ({max(token_lengths)} chars)")
    print(f"Average token length: {sum(token_lengths)/len(token_lengths):.2f} chars")
    print()
    
    # Show first 10 tokens
    print("First 10 tokens:")
    for i, token in enumerate(tokens[:10]):
        print(f"  {i+1}. '{token}'")
    print()
    
    # Unique tokens
    unique_tokens = len(set(tokens))
    print(f"Unique tokens: {unique_tokens}")
    print(f"Token diversity: {unique_tokens/len(tokens):.2%}")


if __name__ == "__main__":
    print("Day 72: Tokenization & Embeddings - Solutions\n")
    
    print("=" * 60)
    print("Exercise 1: Compare Tokenizers")
    print("=" * 60)
    exercise_1_compare_tokenizers()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Subword Analysis")
    print("=" * 60)
    exercise_2_subword_analysis()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Embedding Similarity")
    print("=" * 60)
    exercise_3_embedding_similarity()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Vocabulary Analysis")
    print("=" * 60)
    exercise_4_vocabulary_analysis()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Token Statistics")
    print("=" * 60)
    exercise_5_token_statistics()
