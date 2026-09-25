import crewai.llms.cache as _crewai_cache

_crewai_cache.mark_cache_breakpoint = lambda msg: msg

import os

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from ddgs import DDGS


# ==========================================
# DuckDuckGo Search Tool
# ==========================================

class DuckDuckGoSearchTool(BaseTool):

    name: str = "DuckDuckGo Search"

    description: str = (
        "Search the internet using DuckDuckGo "
        "to find information about a research topic."
    )

    def _run(self, query: str) -> str:

        try:
            results = DDGS().text(
                query,
                max_results=3
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

            return "\n\n".join(output)

        except Exception as e:

            return f"Search error: {str(e)}"


# ==========================================
# Research Agent
# ==========================================

def run_research(topic: str):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing."
        )

    # --------------------------------------
    # Search tool
    # --------------------------------------

    search_tool = DuckDuckGoSearchTool()

    # --------------------------------------
    # Groq LLM
    # --------------------------------------

    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        max_tokens=800
    )

    # --------------------------------------
    # Single CrewAI Agent
    # --------------------------------------

    researcher = Agent(

        role="AI Research Analyst",

        goal=(
            "Research a topic using web search "
            "and create a concise factual report."
        ),

        backstory=(
            "You are an AI research analyst who searches "
            "the web, compares information, and summarizes "
            "reliable sources clearly."
        ),

        tools=[search_tool],

        llm=llm,

        verbose=True,

        allow_delegation=False
    )

    # --------------------------------------
    # Research Task
    # --------------------------------------

    research_task = Task(

        description=f"""
Research this topic:

{topic}

Use DuckDuckGo to find relevant and recent information.

Requirements:

- Search multiple relevant sources.
- Prefer reliable sources.
- Compare important information.
- Do not invent facts.
- Do not invent sources.
- Keep the final report concise.

Write:

# Research Report

## Executive Summary

Give a short summary.

## Key Findings

List the most important findings.

## Analysis

Explain the topic clearly.

## Conclusion

Give a short conclusion.

## Sources

List the URLs you used.
""",

        expected_output=(
            "A concise research report with an executive summary, "
            "key findings, analysis, conclusion, and source URLs."
        ),

        agent=researcher
    )

    # --------------------------------------
    # Crew
    # --------------------------------------

    crew = Crew(

        agents=[researcher],

        tasks=[research_task],

        verbose=True
    )

    # --------------------------------------
    # Run
    # --------------------------------------

    result = crew.kickoff()

    return str(result)
