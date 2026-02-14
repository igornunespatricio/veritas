"""Writer Agent - Produces polished, structured reports."""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.agents.base import BaseAgent
from src.domain.events import ReportWritten, SynthesisCompleted
from src.domain.interfaces import AgentContext


# Define formatting tool for the agent
@tool
def format_report(title: str, content: str, format: str = "markdown") -> str:
    """Format a report into the specified output format.

    Args:
        title: The report title
        content: The report content
        format: Output format (markdown, plain, html)

    Returns:
        Formatted report as JSON string
    """
    return json.dumps({"title": title, "content": content, "format": format})


class WriterAgent(BaseAgent[ReportWritten]):
    """Writer Agent implementation.

    Produces polished, structured reports in various formats
    (markdown, plain text, etc.) from synthesized insights.
    """

    WRITER_SYSTEM_PROMPT = """You are an expert technical writer. Your task is to:
1. Transform research insights into a well-structured report
2. Use clear headings and logical flow
3. Support claims with evidence from the research
4. Adapt tone and format to the requested output style
5. Include proper citations and source references

Write publication-ready content that is accurate and engaging.

IMPORTANT: You must use the format_report tool to output your final report."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ):
        super().__init__(
            name="writer",
            description="Produces polished, structured reports",
            llm_provider=provider,
            llm_model=model,
            llm_temperature=temperature,
            llm_max_tokens=max_tokens,
        )

    async def _run(
        self,
        inputs: dict[str, Any],
        context: AgentContext,
    ) -> ReportWritten:
        """Write a report from synthesis results.

        Uses direct tool calling pattern to ensure structured output.

        Args:
            inputs: Dict with 'synthesis' and 'format'
            context: Agent context with correlation ID

        Returns:
            ReportWritten event with title and content
        """
        synthesis = inputs.get("synthesis")
        report_format = inputs.get("format", "markdown")

        # Runtime type validation for type narrowing
        assert isinstance(
            synthesis, SynthesisCompleted
        ), "synthesis must be SynthesisCompleted"

        insights_text = "\n".join(f"- {insight}" for insight in synthesis.insights)

        contradictions_text = "\n".join(
            f"- {item}" for item in synthesis.resolved_contradictions
        )

        format_instructions = {
            "markdown": "Use Markdown formatting with headers, bullet points, and emphasis.",
            "plain": "Use plain text without any formatting.",
            "html": "Use HTML tags for structure and formatting.",
        }.get(report_format, "Use Markdown formatting.")

        llm = self.llm.llm

        # Check if LLM supports tool calling
        if hasattr(llm, "bind_tools"):
            # Use direct tool calling pattern
            llm_with_tools = llm.bind_tools([format_report])

            messages = [
                SystemMessage(content=self.WRITER_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"Write a comprehensive report based on the following synthesis:\n\n"
                    f"INSIGHTS:\n{insights_text}\n\n"
                    f"RESOLVED CONTRADICTIONS:\n{contradictions_text}\n\n"
                    f"{format_instructions}\n\n"
                    "Use the format_report tool to output your final report with:\n"
                    "- title: descriptive report title about the topic\n"
                    "- content: the full report text about the synthesis insights\n"
                    "- format: the format used (markdown/plain/html)"
                ),
            ]

            response = await llm_with_tools.ainvoke(messages)

            # Check if the model wants to call a tool
            tool_calls = getattr(response, "tool_calls", None)
            if tool_calls:
                # Execute the tool call
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})

                    if tool_name == "format_report":
                        result = format_report.invoke(tool_args)
                        data = json.loads(result)
                        title = data.get("title", "Research Report")
                        report_content = data.get("content", "")
                        fmt = data.get("format", report_format)

                        return ReportWritten.create(
                            title=title,
                            content=report_content,
                            format=fmt,
                            correlation_id=context.correlation_id,
                        )

            # Fallback: parse direct response
            content = (
                response.content if hasattr(response, "content") else str(response)
            )
        else:
            # Fallback for non-tool-calling LLMs
            messages = [
                SystemMessage(content=self.WRITER_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"Write a comprehensive report based on the following synthesis:\n\n"
                    f"INSIGHTS:\n{insights_text}\n\n"
                    f"RESOLVED CONTRADICTIONS:\n{contradictions_text}\n\n"
                    f"{format_instructions}\n\n"
                    "Provide your report in JSON format with:\n"
                    "- title: descriptive report title\n"
                    "- content: the full report text\n"
                    "- format: the format used (markdown/plain/html)"
                ),
            ]

            response = await self.llm.ainvoke(messages)
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

        # Ensure content is a string (handle Ollama list responses)
        if isinstance(content, list):
            content = str(content)
        elif not isinstance(content, str):
            content = str(content)

        # Parse JSON response (fallback or non-tool-calling path)
        try:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_content = content[json_start:json_end]
                data = json.loads(json_content)
                title = data.get("title", "Research Report")
                report_content = data.get("content", content)
                fmt = data.get("format", report_format)
            else:
                title = "Research Report"
                report_content = content
                fmt = report_format
        except json.JSONDecodeError:
            title = "Research Report"
            report_content = content
            fmt = report_format

        return ReportWritten.create(
            title=title,
            content=report_content,
            format=fmt,
            correlation_id=context.correlation_id,
        )

    async def validate_input(self, input: Any) -> bool:
        """Validate input contains synthesis and format."""
        if isinstance(input, dict):
            return "synthesis" in input
        return False

    async def write_report(
        self,
        synthesis: SynthesisCompleted,
        context: AgentContext,
        format: str = "markdown",
    ) -> ReportWritten:
        """Convenience method to write a report."""
        return await self.execute(
            {"synthesis": synthesis, "format": format},
            context,
        )
