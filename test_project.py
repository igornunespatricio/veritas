"""Test script to verify the Veritas multi-agent research workflow."""

import asyncio


from src.orchestration import ResearchWorkflow


async def main():
    """Run a simple test of the research workflow."""
    print("=" * 60)
    print("Veritas Multi-Agent Research Platform - Test")
    print("=" * 60)

    # Initialize the workflow
    # workflow = ResearchWorkflow()
    workflow = ResearchWorkflow(
        llm_provider="openrouter", llm_model="openai/gpt-5-nano"
    )

    # Test topic
    topic = "What is quantum computing and how does it work?"

    print(f"\nTopic: {topic}")
    print("-" * 60)

    # Execute the workflow
    result = await workflow.execute(topic)

    # Report results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"Status: {result.status.value}")

    if result.status.value == "completed":
        print("\n✓ Research completed!")
        print(f"  - Sources found: {len(result.research.sources)}")
        print(f"  - Findings: {len(result.research.findings)}")
        print(f"  - Insights synthesized: {len(result.synthesis.insights)}")
        print(f"  - Review iterations: {result.iterations}")
        print(f"  - Report approved: {result.review.approved}")
        print(f"  - Quality score: {result.review.score:.2f}")

        print("\n" + "-" * 60)
        print("REPORT PREVIEW")
        print("-" * 60)
        print(f"\nTitle: {result.report.title}")
        print("\nContent (first 1000 chars):")
        print(
            result.report.content[:1000] + "..."
            if len(result.report.content) > 1000
            else result.report.content
        )

        if result.review.suggestions:
            print("\n" + "-" * 60)
            print("CRITIC SUGGESTIONS")
            print("-" * 60)
            for i, suggestion in enumerate(result.review.suggestions, 1):
                print(f"{i}. {suggestion}")

    elif result.status.value == "failed":
        print("\n✗ Workflow failed!")
        print(f"  Error: {result.error}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
