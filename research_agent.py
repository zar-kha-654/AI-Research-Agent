import os

import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from ddgs import DDGS


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = "Search the web for information."

    def _run(self, query: str) -> str:
        results = DDGS().text(query, max_results=2)

        if not results:
            return "No results found."

        output = []

        for result in results:
            title = result.get("title", "")
            body = result.get("body", "")[:400]
            url = result.get("href", "")

            output.append(
                f"Title: {title}\n"
                f"Summary: {body}\n"
                f"URL: {url}"
            )

        return "\n\n".join(output)


def run_research(topic):

    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        max_completion_tokens=300,
        temperature=0
    )

    search_tool = DuckDuckGoSearchTool()

    researcher = Agent(
        role="Research Analyst",
        goal="Find accurate information and write a concise report.",
        backstory="You are a research analyst who uses web search.",
        tools=[search_tool],
        llm=llm,
        allow_delegation=False,
        verbose=False
    )

    task = Task(
        description=f"""
Research this topic:

{topic}

Use web search.

Write a SHORT factual report containing:
- Summary
- 3 key findings
- Conclusion
- Sources with URLs

Do not invent information.
""",
        expected_output="A short factual research report with sources.",
        agent=researcher
    )

    crew = Crew(
        agents=[researcher],
        tasks=[task],
        verbose=False
    )

    result = crew.kickoff()

    return str(result)
