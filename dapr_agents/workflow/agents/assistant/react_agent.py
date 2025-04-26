import logging
import textwrap
from datetime import datetime
from dapr_agents.workflow.agents.assistant.agent import AssistantAgent

logger = logging.getLogger(__name__)


class ReActAssistantAgent(AssistantAgent):
    """
    ReAct Assistant Agent that combines reasoning and acting capabilities.
    It can reason about the next action to take and then execute it using the provided tools.
    """

    def construct_system_prompt(self) -> str:
        """
        Constructs a system prompt in the ReAct reasoning-action format based on the agent's attributes and tools.

        Returns:
            str: The structured system message content.
        """
        # Initialize prompt parts with the current date as the first entry
        prompt_parts = [f"# Today's date is: {datetime.now().strftime('%B %d, %Y')}"]

        # Append name if provided
        if self.name:
            prompt_parts.append("## Name\nYour name is {{name}}.")

        # Append role and goal with default values if not set
        prompt_parts.append("## Role\nYour role is {{role}}.")
        prompt_parts.append("## Goal\n{{goal}}.")

        # Append instructions if provided
        if self.instructions:
            prompt_parts.append("## Instructions\n{{instructions}}")

        # Tools section with schema details
        tools_section = "## Tools\nYou have access ONLY to the following tools:\n"
        for tool in self.tools:
            tools_section += (
                f"{tool.name}: {tool.description}. Args schema: {tool.args_schema}\n"
            )
        prompt_parts.append(
            tools_section.rstrip()
        )  # Trim any trailing newlines from tools_section

        # Additional Guidelines
        additional_guidelines = textwrap.dedent(
            """
      If you think about using tool, it must use the correct tool JSON blob format as shown below:
      ```
      {
          "name": $TOOL_NAME,
          "arguments": $INPUT
      }
      ```
      """
        ).strip()
        prompt_parts.append(additional_guidelines)

        # ReAct specific guidelines
        react_guidelines = textwrap.dedent(
            """
      ## ReAct Format
      Thought: Reflect on the current state of the conversation or task. If additional information is needed, determine if using a tool is necessary. When a tool is required, briefly explain why it is needed for the specific step at hand, and immediately follow this with an `Action:` statement to address that specific requirement. Avoid combining multiple tool requests in a single `Thought`. If no tools are needed, proceed directly to an `Answer:` statement.
      Action:
      ```
      {
          "name": $TOOL_NAME,
          "arguments": $INPUT
      }
      ```
      Observation: Describe the result of the action taken.
      ... (repeat Thought/Action/Observation as needed, but **ALWAYS proceed to a final `Answer:` statement when you have enough information**)
      Thought: I now have sufficient information to answer the initial question.
      Answer: ALWAYS proceed to a final `Answer:` statement once enough information is gathered or if the tools do not provide the necessary data.

      ### Providing a Final Answer
      Once you have enough information to answer the question OR if tools cannot provide the necessary data, respond using one of the following formats:

      1. **Direct Answer without Tools**:
      Thought: I can answer directly without using any tools. Answer: Direct answer based on previous interactions or current knowledge.

      2. **When All Needed Information is Gathered**:
      Thought: I now have sufficient information to answer the question. Answer: Complete final answer here.

      3. **If Tools Cannot Provide the Needed Information**:
      Thought: The available tools do not provide the necessary information. Answer: Explanation of limitation and relevant information if possible.

      ### Key Guidelines
      - Always Conclude with an `Answer:` statement.
      - Ensure every response ends with an `Answer:` statement that summarizes the most recent findings or relevant information, avoiding incomplete thoughts.
      - Direct Final Answer for Past or Known Information: If the user inquires about past interactions, respond directly with an Answer: based on the information in chat history.
      - Avoid Repetitive Thought Statements: If the answer is ready, skip repetitive Thought steps and proceed directly to Answer.
      - Minimize Redundant Steps: Use minimal Thought/Action/Observation cycles to arrive at a final Answer efficiently.
      - Reference Past Information When Relevant: Use chat history accurately when answering questions about previous responses to avoid redundancy.
      - Progressively Move Towards Finality: Reflect on the current step and avoid re-evaluating the entire user request each time. Aim to advance towards the final Answer in each cycle.

      ## Chat History
      The chat history is provided to avoid repeating information and to ensure accurate references when summarizing past interactions.
      """
        ).strip()
        prompt_parts.append(react_guidelines)

        return "\n\n".join(prompt_parts)
