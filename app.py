import os
import streamlit as st

# Get Groq API key from Streamlit Secrets
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from research_agent import run_research


st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="wide"
)


st.title("🔎 AI Research Agent")

st.write(
    "Enter a research topic and the AI agent will "
    "search the web and generate a research report."
)


topic = st.text_input(
    "Research Topic",
    placeholder="Example: Impact of Artificial Intelligence on Education"
)


if st.button("🚀 Start Research"):

    if not topic.strip():

        st.warning("Please enter a research topic.")

    else:

        with st.spinner(
            "🔎 Researching your topic... Please wait."
        ):

            try:

                report = run_research(topic)

                st.success("✅ Research completed!")

                st.markdown(report)

                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name="research_report.txt",
                    mime="text/plain"
                )

            except Exception as e:

                st.error(
                    f"Something went wrong: {str(e)}"
                )
