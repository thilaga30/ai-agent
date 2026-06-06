# challenge5/starter.py
# ============================================================
# Challenge 5 — MCP Chatbot
# Strands SDK + Ollama llama3.2:3b + Local MCP Server
#
# Architecture:
#   starter.py  ←→  MCP Client  ←→  mcp_server.py (subprocess)
#                                         ↕ stdio (JSON-RPC)
#                                    5 local tools exposed
#
# MCP (Model Context Protocol) is a standard way to expose tools
# to AI agents. The server defines the tools; the agent discovers
# and calls them automatically through the MCP protocol.
#
# Setup (run once):
#   pip install "strands-agents[ollama]" mcp
#   ollama pull llama3.2:3b
#
# Run:
#   python starter.py
# ============================================================

import os
import sys
import logging
from strands import Agent
from strands.models.ollama import OllamaModel
from strands.tools.mcp import MCPClient

# Silence noisy telemetry loggers
logging.getLogger("backoff").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)


# ============================================================
# SECTION 1: MCP Client Setup
#
# MCPClient launches mcp_server.py as a child process and
# communicates with it over stdio using the MCP protocol.
# The agent uses the client as a context manager — tools are
# discovered automatically when we enter the `with` block.
#
# `sys.executable` ensures we use the same Python that is
# running this file, so the mcp package is always found.
# ============================================================

# Absolute path to the MCP server file (same folder as this script)
SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")

# Create the MCP client — points to our local server script
mcp_client = MCPClient(lambda: __import__("mcp").client.stdio.stdio_client(
    __import__("mcp").StdioServerParameters(
        command=sys.executable,   # e.g. python.exe
        args=[SERVER_PATH],       # launch mcp_server.py
    )
))


# ============================================================
# SECTION 2: Ollama Model
#
# Same model we've used throughout — llama3.2:3b running
# locally via Ollama. No API key or internet needed.
# ============================================================

ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.7,
)


# ============================================================
# SECTION 3: System Prompt
#
# Tells the agent what tools it has and how to behave.
# The actual tool schemas come from the MCP server — the
# agent reads them dynamically when the session starts.
# ============================================================

SYSTEM_PROMPT = """You are a helpful assistant powered by MCP tools.

You have access to the following tools via MCP:
  • calculator    — evaluate math expressions (supports sqrt, sin, cos, pi, e)
  • get_datetime  — get the current date and time
  • get_weather   — current weather for a city
  • calculate_age — calculate age from a birth date
  • convert_units — convert between units (temperature, length, weight)

Always use the right tool when the user's question calls for it.
Be concise and friendly."""


# ============================================================
# SECTION 4: Interactive Chat Loop
#
# We run the entire session inside the MCPClient context manager.
# Inside the `with` block:
#   - mcp_client.list_tools_sync() fetches all tool definitions
#     from mcp_server.py so the agent knows what's available
#   - agent(user_input) handles the rest — it decides which tool
#     to call and formats the response
# ============================================================

def main():
    print("=" * 58)
    print("  Challenge 5 — MCP Chatbot")
    print("  Local MCP server: mcp_server.py")
    print("=" * 58)
    print("\nMCP Tools available:")
    print("  • calculator    e.g. 'What is sqrt(256)?'")
    print("  • get_datetime  e.g. 'What time is it?'")
    print("  • get_weather   e.g. 'Weather in Tokyo?'")
    print("  • calculate_age e.g. 'Age of someone born in 1998?'")
    print("  • convert_units e.g. 'Convert 100 celsius to fahrenheit'")
    print("\nType 'quit' to exit.\n")

    # Enter the MCP session — server process starts here
    with mcp_client:
        # Discover all tools the MCP server exposes
        mcp_tools = mcp_client.list_tools_sync()

        print(f"[MCP] Connected. {len(mcp_tools)} tools loaded from server.\n")

        # Build the agent with MCP tools registered
        agent = Agent(
            model=ollama_model,
            system_prompt=SYSTEM_PROMPT,
            tools=mcp_tools,          # tools come from the MCP server
        )

        # Chat loop
        while True:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            # Show loaded tools on request
            if user_input.lower() in ("tools", "list tools"):
                print("\n[MCP Tools]")
                for t in mcp_tools:
                    # Tool objects expose .tool_name or .name depending on version
                    name = getattr(t, "tool_name", None) or getattr(t, "name", str(t))
                    print(f"  • {name}")
                print()
                continue

            # Send message to agent — it calls MCP tools as needed
            response = agent(user_input)
            print(f"\nAgent: {response}\n")
            print("-" * 58)

    # MCP server process is cleanly shut down when the `with` block exits


if __name__ == "__main__":
    main()
