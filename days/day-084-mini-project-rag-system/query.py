"""Interactive Query Interface for RAG System"""

import sys
from rag_system import RAGSystem
from typing import List, Dict


class QueryInterface:
    """Interactive CLI for RAG system."""
    
    def __init__(self, collection_name: str = "rag_docs"):
        self.rag = RAGSystem(collection_name=collection_name)
        self.history = []
    
    def run(self):
        """Run interactive loop."""
        print("=" * 60)
        print("RAG System - Interactive Query Interface")
        print("=" * 60)
        print("\nCommands:")
        print("  query <text>     - Ask a question")
        print("  search <text>    - Search without generation")
        print("  index <file>     - Index a document file")
        print("  stats            - Show system statistics")
        print("  history          - View query history")
        print("  clear            - Clear all documents")
        print("  help             - Show this help")
        print("  exit             - Quit")
        print("\n" + "=" * 60 + "\n")
        
        while True:
            try:
                user_input = input("> ").strip()
                if not user_input:
                    continue
                
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if command == "exit" or command == "quit":
                    print("\nGoodbye!")
                    break
                elif command == "help":
                    self.show_help()
                elif command == "query":
                    if args:
                        self.handle_query(args)
                    else:
                        print("Usage: query <your question>")
                elif command == "search":
                    if args:
                        self.handle_search(args)
                    else:
                        print("Usage: search <search terms>")
                elif command == "index":
                    if args:
                        self.handle_index(args)
                    else:
                        print("Usage: index <filepath>")
                elif command == "stats":
                    self.show_stats()
                elif command == "history":
                    self.show_history()
                elif command == "clear":
                    self.handle_clear()
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit.")
            except Exception as e:
                print(f"Error: {e}")
    
    def handle_query(self, question: str):
        """Handle query command."""
        print(f"\nQuerying: {question}")
        print("-" * 60)
        
        try:
            result = self.rag.query(question, top_k=3, method="semantic")
            self.history.append({'type': 'query', 'input': question, 'result': result})
            
            print(f"\nAnswer:\n{result['answer']}\n")
            
            if result['sources']:
                print(f"Sources ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'], 1):
                    text = source['text'][:100] + "..." if len(source['text']) > 100 else source['text']
                    print(f"  [{i}] {text}")
            
            print(f"\nConfidence: {result['confidence']:.2f}")
            if result['citations']:
                print(f"Citations used: {result['citations']}")
        
        except Exception as e:
            print(f"Error during query: {e}")
        
        print()
    
    def handle_search(self, query: str):
        """Handle search command."""
        print(f"\nSearching: {query}")
        print("-" * 60)
        
        try:
            results = self.rag.search_only(query, top_k=5, method="semantic")
            self.history.append({'type': 'search', 'input': query, 'results': len(results)})
            
            if results:
                print(f"\nFound {len(results)} results:\n")
                for i, result in enumerate(results, 1):
                    text = result['text'][:150] + "..." if len(result['text']) > 150 else result['text']
                    print(f"{i}. {text}")
                    if 'distance' in result:
                        print(f"   Distance: {result['distance']:.3f}\n")
            else:
                print("No results found.")
        
        except Exception as e:
            print(f"Error during search: {e}")
        
        print()
    
    def handle_index(self, filepath: str):
        """Handle index command."""
        print(f"\nIndexing: {filepath}")
        print("-" * 60)
        
        try:
            result = self.rag.index_file(filepath, metadata={'source': filepath})
            print(f"\nSuccess!")
            print(f"  Documents: {result['documents']}")
            print(f"  Chunks: {result['chunks']}")
            print(f"  Indexed: {result['indexed']}")
        
        except FileNotFoundError:
            print(f"Error: File not found: {filepath}")
        except Exception as e:
            print(f"Error during indexing: {e}")
        
        print()
    
    def show_stats(self):
        """Show system statistics."""
        print("\nSystem Statistics")
        print("-" * 60)
        
        try:
            stats = self.rag.get_stats()
            print(f"Collection: {stats['name']}")
            print(f"Documents: {stats['count']}")
            print(f"Queries: {len([h for h in self.history if h['type'] == 'query'])}")
            print(f"Searches: {len([h for h in self.history if h['type'] == 'search'])}")
        
        except Exception as e:
            print(f"Error getting stats: {e}")
        
        print()
    
    def show_history(self):
        """Show query history."""
        print("\nQuery History")
        print("-" * 60)
        
        if not self.history:
            print("No history yet.")
        else:
            for i, entry in enumerate(self.history[-10:], 1):
                if entry['type'] == 'query':
                    print(f"{i}. Query: {entry['input']}")
                    conf = entry['result'].get('confidence', 0)
                    print(f"   Confidence: {conf:.2f}\n")
                elif entry['type'] == 'search':
                    print(f"{i}. Search: {entry['input']}")
                    print(f"   Results: {entry['results']}\n")
        
        print()
    
    def handle_clear(self):
        """Clear all documents."""
        confirm = input("Are you sure you want to clear all documents? (yes/no): ")
        if confirm.lower() == 'yes':
            try:
                self.rag.clear()
                print("All documents cleared.")
            except Exception as e:
                print(f"Error clearing documents: {e}")
        else:
            print("Cancelled.")
        print()
    
    def show_help(self):
        """Show help message."""
        print("\nAvailable Commands:")
        print("-" * 60)
        print("  query <text>     - Ask a question and get an answer with sources")
        print("  search <text>    - Search for relevant documents without generation")
        print("  index <file>     - Index a document from file (txt, md)")
        print("  stats            - Show system statistics")
        print("  history          - View recent query history")
        print("  clear            - Clear all indexed documents")
        print("  help             - Show this help message")
        print("  exit             - Exit the program")
        print()


def main():
    """Main entry point."""
    collection_name = "rag_docs"
    
    if len(sys.argv) > 1:
        collection_name = sys.argv[1]
    
    interface = QueryInterface(collection_name=collection_name)
    interface.run()


if __name__ == "__main__":
    main()
