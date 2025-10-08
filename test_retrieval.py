"""Test script to see retrieval (RAG without generation) output."""

import sys
from services.rag_service import RAGService
from exceptions import ValidationError, SearchError, DataError

def test_retrieval():
    """Test the retrieval part of RAG without generation."""

    # Sample job queries
    queries = [
        "Python developer with machine learning experience",
        "Full-stack developer with React and Node.js",
        "Data scientist with AI/LLM knowledge",
        "Strong Python skills and familiarity with AI/LLM concepts"
    ]

    print("=" * 80)
    print("SEMANTIC SEARCH RETRIEVAL TEST")
    print("=" * 80)

    try:
        # Initialize the RAG service
        print("\n🔧 Initializing RAG Service (without generation model)...")
        rag_service = RAGService()
        print("✅ RAG Service initialized successfully\n")

        # Test each query
        for i, query in enumerate(queries, 1):
            print(f"\n{'=' * 80}")
            print(f"Test #{i}: {query}")
            print(f"{'=' * 80}")

            try:
                # Search for similar content
                results, scores = rag_service.search_similar(query, k=5)

                print(f"\n🔍 Top 5 Similar Results:")
                print(f"{'─' * 80}")

                if results:
                    for j, result in enumerate(results, 1):
                        print(f"\n#{j} - Score: {result['score']:.4f}")
                        print(f"   Source: {result['source']}")
                        print(f"   Text: {result['text']}")
                else:
                    print("No results found")

            except ValidationError as e:
                print(f"\n❌ Validation Error: {e.message}")
            except SearchError as e:
                print(f"\n❌ Search Error: {e.message}")
            except DataError as e:
                print(f"\n❌ Data Error: {e.message}")
            except Exception as e:
                print(f"\n❌ Unexpected Error: {str(e)}")

        print(f"\n{'=' * 80}")
        print("✅ Test completed!")
        print(f"{'=' * 80}\n")

        # Show service health
        print("\n📊 Service Health Check:")
        health = rag_service.health_check()
        print(f"   Status: {health.get('status', 'unknown')}")
        print(f"   Embedding Model: {health.get('embedding_model', 'N/A')}")
        print(f"   Data Files Valid: {health.get('data_files', False)}")
        print(f"   Embedding Dimension: {health.get('embedding_dimension', 'N/A')}")
        print(f"   Data Size: {health.get('data_size', 0)} items")
        print()

    except Exception as e:
        print(f"\n❌ Failed to initialize: {str(e)}")
        print(f"\nError type: {type(e).__name__}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_retrieval()

