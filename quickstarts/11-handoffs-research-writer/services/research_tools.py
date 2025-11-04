"""
Fake tools for the research and writing workflow.
These simulate web search and document management operations.
"""
from dapr_agents import tool
from pydantic import BaseModel, Field


class SearchWebSchema(BaseModel):
    query: str = Field(description="The search query to look up information")


@tool(args_model=SearchWebSchema)
def search_web(query: str) -> str:
    """Search the web for information on a given topic. Returns simulated search results."""
    # Simulated search results based on common queries
    search_results = {
        "artificial intelligence": """
        Recent developments in artificial intelligence show significant progress in large language models,
        with improvements in reasoning capabilities and multimodal understanding. Key trends include:
        - Enhanced reasoning and planning abilities
        - Better context understanding and memory
        - Improved safety and alignment techniques
        - Integration with specialized tools and APIs
        """,
        "climate change": """
        Climate change research indicates accelerating impacts globally:
        - Rising global temperatures and extreme weather events
        - Ocean acidification and ecosystem disruption
        - Renewable energy adoption increasing worldwide
        - International cooperation on carbon reduction goals
        """,
        "quantum computing": """
        Quantum computing advances are showing promising results:
        - Error correction techniques improving qubit stability
        - Hybrid quantum-classical algorithms gaining traction
        - Commercial applications in optimization and simulation
        - Major tech companies investing in quantum infrastructure
        """,
        "default": """
        General information about the topic suggests ongoing research and development.
        Multiple perspectives exist, with experts highlighting various aspects of the subject.
        Recent studies show interesting patterns and potential future directions.
        Further investigation would provide more specific insights.
        """
    }
    
    # Find matching result or use default
    query_lower = query.lower()
    for key, result in search_results.items():
        if key in query_lower:
            return f"Search results for '{query}':\n{result.strip()}"
    
    return f"Search results for '{query}':\n{search_results['default'].strip()}"


class RecordNotesSchema(BaseModel):
    notes: str = Field(description="The notes to record")
    notes_title: str = Field(description="The title for the notes")


@tool(args_model=RecordNotesSchema)
def record_notes(notes: str, notes_title: str) -> str:
    """Record notes on a given topic. Useful for saving research findings."""
    # In a real implementation, this would save to a database or file
    # For demo purposes, we just acknowledge the recording
    return f"Notes recorded successfully under title: '{notes_title}'"


class WriteReportSchema(BaseModel):
    report_content: str = Field(description="The markdown formatted report content")


@tool(args_model=WriteReportSchema)
def write_report(report_content: str) -> str:
    """Write a report on a given topic. Your input should be markdown formatted."""
    # In a real implementation, this would save the report
    # For demo purposes, we just acknowledge the report creation
    word_count = len(report_content.split())
    return f"Report written successfully. Contains {word_count} words."


class ReviewReportSchema(BaseModel):
    review: str = Field(description="The review feedback for the report")


@tool(args_model=ReviewReportSchema)
def review_report(review: str) -> str:
    """Review a report and provide feedback. Your input should be constructive feedback."""
    # In a real implementation, this would save the review
    # For demo purposes, we just acknowledge the review
    return "Review submitted successfully. Feedback recorded."


# Tool collections for each agent
research_tools = [search_web, record_notes]
writer_tools = [write_report]
reviewer_tools = [review_report]
