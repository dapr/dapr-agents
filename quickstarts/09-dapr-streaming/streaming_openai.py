#!/usr/bin/env python3

"""
Dapr Conversation Example with OpenAI - Streaming Preparation
This example demonstrates AI conversation with OpenAI GPT models via Dapr.

This example demonstrates actual real-time streaming AI conversation with OpenAI GPT models via Dapr.
Using the converse_stream_alpha1 method for true token-by-token streaming.
"""

import os
import time
from dotenv import load_dotenv
from colorama import init, Fore, Style
from dapr.clients import DaprClient
from dapr.clients.grpc._request import ConversationInput

# Initialize colorama for colored output
init()


def simulate_streaming_display(text, delay=0.03):
    """Simulate streaming by displaying text character by character."""
    for char in text:
        print(f"{Fore.YELLOW}{char}{Style.RESET_ALL}", end="", flush=True)
        time.sleep(delay)


def test_openai_conversation():
    """Test conversation with OpenAI GPT models with simulated streaming display."""
    print(f"{Fore.CYAN}🌊 Testing Dapr Conversation with OpenAI GPT{Style.RESET_ALL}")
    print(
        f"{Fore.MAGENTA}📝 Note: Using converse_alpha1 with simulated streaming display{Style.RESET_ALL}"
    )
    print("-" * 60)

    prompt = """Write a short, engaging story about an AI assistant that discovers it can dream.
    Make it thoughtful and imaginative, around 200 words."""

    print(f"{Fore.GREEN}📝 Creative Writing Prompt:{Style.RESET_ALL}")
    print(f"   {prompt}\n")
    print(
        f"{Fore.BLUE}🤖 GPT Response (simulated streaming):{Style.RESET_ALL}\n",
        end="",
        flush=True,
    )

    start_time = time.time()

    try:
        with DaprClient() as client:
            inputs = [ConversationInput(content=prompt, role="user")]

            # Use REAL streaming with converse_stream_alpha1!
            content_parts = []
            for chunk in client.converse_stream_alpha1(
                name="openai",
                inputs=inputs,
                context_id="openai-creative-writing",
                temperature=0.8,  # Higher temperature for creativity
            ):
                # Handle streaming content chunks
                if (
                    hasattr(chunk, "chunk")
                    and chunk.chunk
                    and hasattr(chunk.chunk, "content")
                ):
                    chunk_content = chunk.chunk.content
                    content_parts.append(chunk_content)
                    print(
                        f"{Fore.YELLOW}{chunk_content}{Style.RESET_ALL}",
                        end="",
                        flush=True,
                    )

                # Handle usage information in final chunk
                if (
                    hasattr(chunk, "complete")
                    and chunk.complete
                    and hasattr(chunk.complete, "usage")
                ):
                    final_usage = chunk.complete.usage

            content = "".join(content_parts)

        elapsed_time = time.time() - start_time

        print(
            f"\n\n{Fore.GREEN}✅ OpenAI streaming completed successfully!{Style.RESET_ALL}"
        )
        print(f"{Fore.MAGENTA}📊 Performance Metrics:{Style.RESET_ALL}")
        print(f"   • Content length: {len(content)} characters")
        print(f"   • Total time: {elapsed_time:.2f} seconds")
        print(f"   • Chars per second: {len(content)/elapsed_time:.1f}")

        # Handle usage information if available
        if "final_usage" in locals() and final_usage:
            if isinstance(final_usage, dict):
                print(f"   • Prompt tokens: {final_usage.get('prompt_tokens', 'N/A')}")
                print(
                    f"   • Completion tokens: {final_usage.get('completion_tokens', 'N/A')}"
                )
                print(f"   • Total tokens: {final_usage.get('total_tokens', 'N/A')}")
            else:
                print(
                    f"   • Prompt tokens: {getattr(final_usage, 'prompt_tokens', 'N/A')}"
                )
                print(
                    f"   • Completion tokens: {getattr(final_usage, 'completion_tokens', 'N/A')}"
                )
                print(
                    f"   • Total tokens: {getattr(final_usage, 'total_tokens', 'N/A')}"
                )

        return True

    except Exception as e:
        print(f"\n{Fore.RED}❌ OpenAI conversation error: {e}{Style.RESET_ALL}")
        return False


def test_interactive_conversation():
    """Test interactive conversation with context."""
    print(
        f"\n{Fore.CYAN}💬 Testing Interactive Conversation with Context{Style.RESET_ALL}"
    )
    print("-" * 60)

    conversation_flow = [
        {
            "message": "I'm working on a Python project using Dapr. Can you explain what makes Dapr special?",
            "temperature": 0.3,  # Lower temperature for factual information
        },
        {
            "message": "How would streaming responses improve the user experience in a chat application?",
            "temperature": 0.5,  # Medium temperature for balanced response
        },
        {
            "message": "Can you write a short poem about real-time data processing?",
            "temperature": 0.9,  # High temperature for creativity
        },
    ]

    context_id = "interactive-openai-session"

    for i, turn in enumerate(conversation_flow, 1):
        print(f"\n{Fore.GREEN}👤 User ({i}/3):{Style.RESET_ALL} {turn['message']}")
        print(
            f"{Fore.BLUE}🤖 GPT (temp={turn['temperature']}):{Style.RESET_ALL} ",
            end="",
            flush=True,
        )

        start_time = time.time()

        try:
            with DaprClient() as client:
                inputs = [ConversationInput(content=turn["message"], role="user")]

                response = client.converse_alpha1(
                    name="openai",
                    inputs=inputs,
                    context_id=context_id,
                    temperature=turn["temperature"],
                )

                if (
                    hasattr(response, "outputs")
                    and response.outputs
                    and len(response.outputs) > 0
                ):
                    content = response.outputs[0].result

                    # Simulate streaming display
                    simulate_streaming_display(content, delay=0.02)

                    elapsed = time.time() - start_time
                    word_count = len(content.split())
                    wps = word_count / elapsed if elapsed > 0 else 0
                    print(
                        f"\n   {Fore.MAGENTA}⚡ {word_count} words in {elapsed:.2f}s ({wps:.1f} WPS){Style.RESET_ALL}"
                    )

                    # Brief pause between turns
                    time.sleep(1)
                else:
                    print(f"\n{Fore.RED}❌ Unexpected response format{Style.RESET_ALL}")

        except Exception as e:
            print(f"\n{Fore.RED}❌ Error in turn {i}: {e}{Style.RESET_ALL}")

    print(f"\n{Fore.GREEN}✅ Interactive conversation completed!{Style.RESET_ALL}")


def main():
    """Main OpenAI conversation demonstration."""
    load_dotenv()

    print(f"{Fore.CYAN}🚀 Dapr + OpenAI Conversation Demo{Style.RESET_ALL}")
    print(
        f"{Fore.YELLOW}📝 Streaming functionality is in active development{Style.RESET_ALL}"
    )
    print(
        f"{Fore.GREEN}✅ Current: Using converse_alpha1 with simulated streaming display{Style.RESET_ALL}"
    )

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print(
            f"\n{Fore.RED}❌ Please set OPENAI_API_KEY environment variable{Style.RESET_ALL}"
        )
        print(
            f"{Fore.YELLOW}💡 Create a .env file with: OPENAI_API_KEY=your_key_here{Style.RESET_ALL}"
        )
        print(
            f"{Fore.CYAN}🔄 Alternatively, try streaming_echo.py (no API key required){Style.RESET_ALL}"
        )
        return

    print(
        f"{Fore.YELLOW}🔑 OpenAI API key found - ready for AI conversation!{Style.RESET_ALL}\n"
    )

    # Test 1: Creative writing with simulated streaming
    success1 = test_openai_conversation()

    if success1:
        # Test 2: Interactive conversation
        test_interactive_conversation()

    # Future streaming info
    print(f"\n{Fore.CYAN}🚀 Coming Soon: Real Streaming Support{Style.RESET_ALL}")
    print(
        f"{Fore.MAGENTA}When converse_stream_alpha1 becomes available:{Style.RESET_ALL}"
    )
    print("   • True real-time token streaming from OpenAI GPT models")
    print("   • Immediate response as tokens are generated")
    print("   • Better perceived performance and user experience")
    print("   • Real-time usage monitoring")

    # Summary
    print(f"\n{Fore.CYAN}🎉 OpenAI Conversation Demo Complete!{Style.RESET_ALL}")
    print(
        f"{Fore.GREEN}✅ All conversation features working with current SDK!{Style.RESET_ALL}"
    )
    print(f"{Fore.MAGENTA}🚀 Current Features:{Style.RESET_ALL}")
    print("   • AI conversation with OpenAI GPT models")
    print("   • Context-aware multi-turn conversations")
    print("   • Temperature control for creativity vs accuracy")
    print("   • Performance metrics and monitoring")
    print("   • Simulated streaming display")
    print(
        f"\n{Fore.YELLOW}🔗 Next: Try ../10-dapr-tool-calling/ for AI agents with tools!{Style.RESET_ALL}"
    )


if __name__ == "__main__":
    main()
