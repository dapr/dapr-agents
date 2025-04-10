import json
import logging
import textwrap
from datetime import datetime

import regex
from pydantic import ConfigDict, Field

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from dapr_agents.agent import AgentBase
from dapr_agents.tool import AgentTool
from dapr_agents.types import AgentError, AssistantMessage, ChatCompletion

logger = logging.getLogger(__name__)

class ReActAgent(AgentBase):
    """
    Agent implementing the ReAct (Reasoning-Action) framework for dynamic, few-shot problem-solving by leveraging
    contextual reasoning, actions, and observations in a conversation flow.
    """

    stop_at_token: List[str] = Field(default=["\nObservation:"], description="Token(s) signaling the LLM to stop generation.")
    tools: List[Union[AgentTool, Callable]] = Field(default_factory=list, description="Tools available for the agent, including final_answer.")
    template_format: Literal["f-string", "jinja2"] = Field(default="jinja2", description="The format used for rendering the prompt template.")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
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
            tools_section += f"{tool.name}: {tool.description}. Args schema: {tool.args_schema}\n"
        prompt_parts.append(tools_section.rstrip())  # Trim any trailing newlines from tools_section

        # Additional Guidelines
        additional_guidelines = textwrap.dedent("""
        If you think about using tool, it must use the correct tool JSON blob format as shown below:
        ```
        {
            "name": $TOOL_NAME,
            "arguments": $INPUT
        }
        ```
        """).strip()
        prompt_parts.append(additional_guidelines)

        # ReAct specific guidelines
        react_guidelines = textwrap.dedent("""
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
        """).strip()
        prompt_parts.append(react_guidelines)

        return "\n\n".join(prompt_parts)
    
    async def run(self, input_data: Optional[Union[str, Dict[str, Any]]] = None) -> Any:
        """
        Runs the agent in a ReAct-style loop until it generates a final answer or reaches max iterations.

        Args:
            input_data (Optional[Union[str, Dict[str, Any]]]): Initial task or message input.

        Returns:
            Any: The agent's final answer.

        Raises:
            AgentError: If LLM fails or tool execution encounters issues.
        """
        logger.debug(f"Agent run started with input: {input_data or 'Using memory context'}")

        # Format messages; construct_messages already includes chat history.
        messages = self.construct_messages(input_data or {})
        user_message = self.get_last_user_message(messages)

        # Add the new user message to memory only if input_data is provided and user message exists.
        if input_data and user_message:
            self.memory.add_message(user_message)
        
        # Always print the last user message for context, even if no input_data is provided
        if user_message:
            self.text_formatter.print_message(user_message)

        # Get Tool Names to validate tool selection
        available_tools = self.tool_executor.get_tool_names()

        # Initialize react_loop for iterative reasoning
        react_loop = ""

        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations} started.")

            # Check if "react_loop" is already a variable in the template
            if "react_loop" in self.prompt_template.input_variables:
                # If "react_loop" exists as a variable, construct messages dynamically
                iteration_messages = self.construct_messages({"react_loop": react_loop})
            else:
                # Create a fresh copy of original_messages for this iteration
                iteration_messages = [msg.copy() for msg in messages]

                # Append react_loop to the last message (user or otherwise)
                for msg in reversed(iteration_messages):
                    if msg["role"] == "user":
                        msg["content"] += f"\n{react_loop}"
                        break
                else:
                    # Append react_loop to the last message if no user message is found
                    logger.warning("No user message found in the current messages; appending react_loop to the last message.")
                    iteration_messages[-1]["content"] += f"\n{react_loop}"  # Append react_loop to the last message

            try:
                response: ChatCompletion = self.llm.generate(
                    messages=iteration_messages, stop=self.stop_at_token
                )

                # Parse response into thought, action, and potential final answer
                thought_action, action, final_answer = self.parse_response(response)

                # Print Thought immediately
                self.text_formatter.print_react_part("Thought", thought_action)

                if final_answer:
                    assistant_final = AssistantMessage(final_answer)
                    self.memory.add_message(assistant_final)
                    self.text_formatter.print_separator()
                    self.text_formatter.print_message(assistant_final, include_separator=False)
                    logger.info("Agent provided a direct final answer.")
                    return final_answer

                # If there's no action, update the loop and continue reasoning
                if not action:
                    logger.info("No action specified; continuing with further reasoning.")
                    react_loop += f"Thought:{thought_action}\n"
                    continue  # Proceed to the next iteration

                action_name = action["name"]
                action_args = action["arguments"]

                # Print Action
                self.text_formatter.print_react_part("Action", json.dumps(action))

                if action_name not in available_tools:
                    raise AgentError(f"Unknown tool specified: {action_name}")

                logger.info(f"Executing {action_name} with arguments {action_args}")
                result = await self.tool_executor.run_tool(action_name, **action_args)

                # Print Observation
                self.text_formatter.print_react_part("Observation", result)
                react_loop += f"Thought:{thought_action}\nAction:{json.dumps(action)}\nObservation:{result}\n"

            except Exception as e:
                logger.error(f"Error during ReAct agent loop: {e}")
                raise AgentError(f"ReActAgent failed: {e}") from e

        logger.info("Max iterations reached. Agent has stopped.")
    
    
    def parse_response(self, response: ChatCompletion) -> Tuple[str, Optional[dict], Optional[str]]:
        """
        Parses a ReAct-style LLM response into a Thought, optional Action (JSON blob), and optional Final Answer.

        Args:
            response (ChatCompletion): The LLM response object containing the message content.

        Returns:
            Tuple[str, Optional[dict], Optional[str]]:
                - Thought string.
                - Parsed Action dictionary, if present.
                - Final Answer string, if present.
        """
        pattern = r'\{(?:[^{}]|(?R))*\}'  # Recursive pattern to match nested JSON blobs
        content = response.get_content()

        # Compile reusable regex patterns
        action_split_regex = regex.compile(r'action:\s*', flags=regex.IGNORECASE)
        final_answer_regex = regex.compile(r'answer:\s*(.*)', flags=regex.IGNORECASE | regex.DOTALL)
        thought_label_regex = regex.compile(r'thought:\s*', flags=regex.IGNORECASE)

        # Strip leading "Thought:" labels (they get repeated a lot)
        content = thought_label_regex.sub('', content).strip()

        # Check if there's a final answer present
        if final_match := final_answer_regex.search(content):
            final_answer = final_match.group(1).strip()
            logger.debug(f"[parse_response] Final answer detected: {final_answer}")
            return content, None, final_answer

        # Split on first "Action:" to separate Thought and Action
        if action_split_regex.search(content):
            thought_part, action_block = action_split_regex.split(content, 1)
            thought_part = thought_part.strip()
            logger.debug(f"[parse_response] Thought extracted: {thought_part}")
            logger.debug(f"[parse_response] Action block to parse: {action_block.strip()}")
        else:
            logger.debug(f"[parse_response] No action or answer found. Returning content as Thought: {content}")
            return content, None, None

        # Attempt to extract the first valid JSON blob from the action block
        for match in regex.finditer(pattern, action_block, flags=regex.DOTALL):
            try:
                action_dict = json.loads(match.group())
                logger.debug(f"[parse_response] Successfully parsed action: {action_dict}")
                return thought_part, action_dict, None
            except json.JSONDecodeError as e:
                logger.debug(f"[parse_response] Failed to parse action JSON blob: {match.group()} — {e}")
                continue

        logger.debug(f"[parse_response] No valid action JSON found. Returning Thought only.")
        return thought_part, None, None
    
    async def run_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """
        Executes a tool by name, resolving async or sync tools automatically.

        Args:
            tool_name (str): The name of the registered tool.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Any: The tool result.

        Raises:
            AgentError: If execution fails.
        """
        try:
            return await self.tool_executor.run_tool(tool_name, *args, **kwargs)
        except Exception as e:
            logger.error(f"Failed to run tool '{tool_name}' via ReActAgent: {e}")
            raise AgentError(f"Error running tool '{tool_name}': {e}") from e