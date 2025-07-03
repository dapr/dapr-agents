#!/usr/bin/env python3

"""
Dapr Conversation Example with Echo Component
This example demonstrates basic LLM conversation using the echo component.

Note: Streaming functionality (converse_stream_alpha1) is currently in development
and will be available in future versions of the Python SDK.
"""

import time
from dotenv import load_dotenv
from colorama import init, Fore, Style
from dapr_agents.llm import DaprChatClient

# Initialize colorama for colored output
init()


def main():
    """Main conversation example with echo component."""
    load_dotenv()

    print(
        f"{Fore.CYAN}🚀 Testing Dapr Conversation with Echo Component{Style.RESET_ALL}"
    )
    print(
        f"{Fore.YELLOW}This example demonstrates LLM conversation without requiring API keys{Style.RESET_ALL}"
    )
    print(
        f"{Fore.MAGENTA}📝 Note: Streaming support is in development and coming soon!{Style.RESET_ALL}\n"
    )

    # Initialize the Dapr chat client
    llm = DaprChatClient()

    # Test basic conversation
    prompt = "Tell me a story about a brave knight on a quest to save the kingdom."

    print(f"{Fore.GREEN}📝 Prompt:{Style.RESET_ALL} {prompt}\n")
    print(f"{Fore.BLUE}Response:{Style.RESET_ALL}")

    try:
        # Generate response (non-streaming for now)
        start_time = time.time()

        response = llm.generate(
            messages=[{"role": "user", "content": prompt}],
            stream=False,  # Streaming not yet available
            llm_component="echo",
        )

        end_time = time.time()

        # Extract and display the response
        if hasattr(response, "choices") and response.choices:
            content = response.choices[0].message.content
            print(content)
        elif isinstance(response, dict) and "outputs" in response:
            # Handle the raw response format from echo component
            output = response["outputs"][0]
            content = output.get("result", "No content found")
            print(content)
        else:
            print("Response received but format is unexpected")
            print(f"Response type: {type(response)}")
            print(f"Response: {response}")

        print(f"\n{Fore.MAGENTA}✅ Conversation completed!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📊 Stats:{Style.RESET_ALL}")
        print(f"   • Response time: {end_time - start_time:.2f} seconds")
        print("   • Component used: echo")

    except Exception as e:
        print(f"\n{Fore.RED}❌ Error during conversation: {e}{Style.RESET_ALL}")
        print(
            f"{Fore.YELLOW}💡 Make sure Dapr sidecar is running with echo component{Style.RESET_ALL}"
        )
        return

    # Test conversation with context
    print(f"\n{Fore.CYAN}🔄 Testing conversation with context{Style.RESET_ALL}")

    conversation = [
        {"role": "user", "content": "What's your favorite color?"},
        {
            "role": "assistant",
            "content": "I enjoy discussing the concept of blue - it represents depth and tranquility.",
        },
        {"role": "user", "content": "Why do you find that color appealing?"},
    ]

    print(f"{Fore.GREEN}📝 Multi-turn conversation:{Style.RESET_ALL}")
    for msg in conversation:
        role_color = Fore.CYAN if msg["role"] == "user" else Fore.MAGENTA
        print(
            f"   {role_color}{msg['role'].title()}:{Style.RESET_ALL} {msg['content']}"
        )

    print(f"\n{Fore.BLUE}Response:{Style.RESET_ALL}")

    try:
        start_time = time.time()

        response = llm.generate(
            messages=conversation, stream=False, llm_component="echo"
        )

        end_time = time.time()

        # Extract and display the response
        if hasattr(response, "choices") and response.choices:
            content = response.choices[0].message.content
            print(content)
        elif isinstance(response, dict) and "outputs" in response:
            # Handle the raw response format from echo component
            output = response["outputs"][0]
            content = output.get("result", "No content found")
            print(content)

        print(f"\n{Fore.MAGENTA}✅ Context conversation completed!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📊 Stats:{Style.RESET_ALL}")
        print(f"   • Response time: {end_time - start_time:.2f} seconds")

    except Exception as e:
        print(f"\n{Fore.RED}❌ Error during context conversation: {e}{Style.RESET_ALL}")

    # Show what's coming
    print(f"\n{Fore.CYAN}🚀 Coming Soon: Streaming Support{Style.RESET_ALL}")
    print(
        f"{Fore.YELLOW}When streaming is available, you'll be able to:{Style.RESET_ALL}"
    )
    print("   • See responses as they're generated in real-time")
    print("   • Get faster perceived response times")
    print("   • Build more interactive chat experiences")
    print("   • Monitor token usage in real-time")

    print(
        f"\n{Fore.GREEN}✅ Example completed! Try the OpenAI example for more advanced features.{Style.RESET_ALL}"
    )


if __name__ == "__main__":
    main()
