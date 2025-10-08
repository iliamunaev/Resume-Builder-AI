"""Test script to see skills generation output."""

import sys
from services.skills_service import SkillsService
from exceptions import ValidationError, ModelError, DataError

def test_skills_generation():
    """Test the skills generation with sample queries."""

    # Sample job queries
    queries = [
        "Python developer with machine learning experience",
        "Full-stack developer with React and Node.js",
        "Data scientist with AI/LLM knowledge",
        "Strong Python skills and familiarity with AI/LLM concepts"
    ]

    print("=" * 80)
    print("SKILLS GENERATION TEST")
    print("=" * 80)

    try:
        # Initialize the skills service
        print("\n🔧 Initializing Skills Service...")
        skills_service = SkillsService()
        print("✅ Skills Service initialized successfully\n")

        # Test each query
        for i, query in enumerate(queries, 1):
            print(f"\n{'=' * 80}")
            print(f"Test #{i}: {query}")
            print(f"{'=' * 80}")

            try:
                # Generate skills
                skills_section = skills_service.generate_skills_section(
                    job_requirements=query,
                    max_skills=3
                )

                print(f"\n📝 Generated Skills Section:")
                print(skills_section)

                # Extract individual skills
                skills = skills_service.extract_skills_from_text(skills_section, limit=3)
                print(f"\n📋 Extracted Skills List:")
                for skill in skills:
                    print(f"  • {skill}")

            except ValidationError as e:
                print(f"\n❌ Validation Error: {e.message}")
            except ModelError as e:
                print(f"\n❌ Model Error: {e.message}")
            except DataError as e:
                print(f"\n❌ Data Error: {e.message}")
            except Exception as e:
                print(f"\n❌ Unexpected Error: {str(e)}")

        print(f"\n{'=' * 80}")
        print("✅ Test completed!")
        print(f"{'=' * 80}\n")

    except Exception as e:
        print(f"\n❌ Failed to initialize: {str(e)}")
        print(f"\nError type: {type(e).__name__}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_skills_generation()

