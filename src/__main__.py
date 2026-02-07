"""Entry point for running Veritas as a package."""

import asyncio
import sys

from .orchestration import ResearchWorkflow


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: veritas <research_topic>")
        print("Example: veritas 'What is machine learning?'")
        sys.exit(1)

    topic = sys.argv[1]
    asyncio.run(run_research(topic))


async def run_research(topic: str):
    """Run research workflow."""
    print(f"Researching: {topic}")
    workflow = ResearchWorkflow()
    result = await workflow.execute(topic)

    if result.status.value == "completed":
        print("\n" + "=" * 60)
        print(result.report.title)
        print("=" * 60)
        print(result.report.content)
    else:
        print(f"Error: {result.error}")


if __name__ == "__main__":
    main()
