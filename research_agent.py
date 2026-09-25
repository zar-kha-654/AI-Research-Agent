import crewai.llms.cache as _crewai_cache

_crewai_cache.mark_cache_breakpoint = lambda msg: msg

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from ddgs import DDGS
import os


class DuckDuckGoSearchTool(BaseTool):

    name: str = "DuckDuckGo Search"

    description: str = (
        "Search the internet using DuckDuckGo to find "
        "current and relevant information about a research topic."
    )

    def _run(self, query: str) -> str:

        try:
            results = DDGS().text(
                query,
                max_results=5
            )

            if not results:
                return "No search results found."

            output = []

            for result in results:

                title = result.get("title", "")
                body = result.get("body", "")
                url = result.get("href", "")

                output.append(
                    f"TITLE: {title}\n"
                    f"SUMMARY: {body}\n"
                    f"URL: {url}\n"
                )

            return "\n".join(output)

        except Exception as e:
            return f"Search error: {str(e)}"


def run_research(topic: str):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    search_tool = DuckDuckGoSearchTool()

    llm = LLM(
    model="groq/openai/gpt-oss-120b",
    api_key=api_key,
    max_tokens=800
)

    researcher = Agent(

        role="AI Research Analyst",

        goal=(
            "Research the user's topic using web sources "
            "and produce an accurate and well-structured report."
        ),

        backstory=(
            "You are an experienced AI research analyst. "
            "You search for relevant information, compare sources, "
            "identify important facts, and explain complex topics "
            "clearly."
        ),

        tools=[search_tool],

        llm=llm,

        verbose=True,

        allow_delegation=False
    )

    research_task = Task(

        description=f"""
Research this topic:

{topic}

Use DuckDuckGo to find relevant and recent information.

Requirements:
- Search multiple relevant sources.
- Prefer reliable sources.
- Compare important information.
- Do not invent facts or sources.
- Keep the final report concise.

Write:

# Research Report

## Executive Summary
A short summary.

## Key Findings
The most important findings.

## Analysis
Explain the topic clearly.

## Conclusion
A short conclusion.

## Sources
List the URLs you used.
"""
        Make the report clear and beginner-friendly.
        """,

        expected_output=(
            "A complete research report containing an executive summary, "
            "introduction, key findings, detailed analysis, benefits, "
            "limitations, conclusion, and source URLs."
        ),

        agent=researcher
    )

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True
    )

    result = crew.kickoff()

    return str(result)
