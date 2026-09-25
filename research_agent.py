import os

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from ddgs import DDGS


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = "Search the web using DuckDuckGo."

    def _run(self, query: str) -> str:
        results = DDGS().text(query, max_results=3)

        if not results:
            return "No results found."

        text = ""

        for result in results:
            text += (
                f"Title: {result.get('title', '')}\n"
                f"Summary: {result.get('body', '')}\n"
                f"URL: {result.get('href', '')}\n\n"
            )

        return text


def run_research(topic):

    api_key = os.getenv("GROQ_API_KEY")

    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        max_tokens=500
    )

    search_tool = DuckDuckGoSearchTool()

    researcher = Agent(
        role="Research Analyst",
        goal="Research the topic and create a factual report.",
        backstory="You are a research analyst who searches the web.",
        tools=[search_tool],
        llm=llm,
        allow_delegation=False
    )

    task = Task(
        description=f"""
Research this topic:

{topic}

Search the web and write a concise report.

Include:

1. Executive Summary
2. Key Findings
3. Analysis
4. Conclusion
5. Sources with URLs

Do not invent facts or sources.
""",
        expected_output="A concise research report with sources.",
        agent=researcher
    )

    crew = Crew(
        agents=[researcher],
        tasks=[task]
    )

    result = crew.kickoff()

    return str(result)
