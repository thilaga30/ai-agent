# ============================================================
# Challenge 3 — Memory Agent
# Strands SDK + Ollama llama3.2:3b + Persistent Memory
#
# Memory is stored in memory_store/memories.json on disk.
# Everything is local — no API keys, no cloud.
# ============================================================

import os
import json
from strands import Agent
from strands.models.ollama import OllamaModel


# ================================================================
# SECTION 1 — Simple persistent memory store (JSON file)
# ================================================================
# We store memories as a plain JSON file.
# This is reliable, transparent, and survives restarts.
# Format: { "nithya": ["fact1", "fact2", ...], ... }

# Path to the memory file — lives next to this script
MEMORY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "memory_store",
    "memories.json"
)

# The user whose memories we are tracking
USER_ID = "nithya"


def load_memories() -> dict:
    """Load all memories from disk. Returns empty dict if file doesn't exist."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}


def save_memories(memories: dict) -> None:
    """Save all memories to disk, creating the folder if needed."""
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump(memories, f, indent=2)


def add_memory(fact: str) -> None:
    """Add a single fact to the current user's memory list."""
    memories = load_memories()
    user_memories = memories.get(USER_ID, [])
    # Avoid storing duplicates
    if fact not in user_memories:
        user_memories.append(fact)
    memories[USER_ID] = user_memories
    save_memories(memories)


def get_all_user_memories() -> list:
    """Return all stored memory facts for the current user."""
    memories = load_memories()
    return memories.get(USER_ID, [])


def clear_user_memories() -> None:
    """Delete all memories for the current user."""
    memories = load_memories()
    memories[USER_ID] = []
    save_memories(memories)


def format_memories_for_prompt(facts: list) -> str:
    """Format the memory list as a readable string for the system prompt."""
    if not facts:
        return "No memories stored yet."
    return "\n".join(f"  - {fact}" for fact in facts)


# ================================================================
# SECTION 2 — Memory extraction using the LLM
# ================================================================
# We use a lightweight Strands agent just to extract facts from
# what the user says. This keeps the main agent clean.

def extract_facts(user_message: str) -> list:
    """
    Ask the LLM to pull out any personal facts from the user's message.
    Returns a list of short fact strings, or empty list if none found.
    """
    # Create a one-shot extractor agent
    extractor = Agent(
        model=OllamaModel(model_id="llama3.2:3b", host="http://localhost:11434"),
        system_prompt=(
            "You are a fact extractor. "
            "Given a message, extract any personal facts the user is sharing about themselves. "
            "Return ONLY a JSON array of short fact strings. "
            "Example input : 'My name is Thilaga and I love Python' "
            "Example output: [\"Name is Thilaga\", \"Loves Python\"] "
            "If there are no personal facts, return: [] "
            "Return ONLY the JSON array, nothing else."
        ),
    )

    try:
        raw = str(extractor(user_message)).strip()

        # Find the JSON array in the response
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return []

        facts = json.loads(raw[start:end])
        return [f for f in facts if isinstance(f, str) and f.strip()]

    except Exception:
        # If extraction fails for any reason, just skip silently
        return []


# ================================================================
# SECTION 3 — Main Strands agent with memory-aware system prompt
# ================================================================

model = OllamaModel(
    model_id="llama3.2:3b",
    host="http://localhost:11434",
)

# {memory_context} is replaced with the user's stored memories
# before every single message, so the agent always knows the user.
SYSTEM_PROMPT = """You are a helpful, friendly assistant with a persistent memory.

Here is what you remember about the user:
--- MEMORY ---
{memory_context}
--------------

Use this information to personalise your responses.
Greet the user by name if you know it.
When the user shares new personal facts, acknowledge them naturally.
Be warm, concise, and helpful."""

agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT.format(
        memory_context=format_memories_for_prompt(get_all_user_memories())
    ),
)


# ================================================================
# SECTION 4 — One full memory-aware chat turn
# ================================================================

def chat(user_input: str) -> str:
    """
    Full memory turn:
    1. Extract any new facts from the user's message → store them
    2. Load all memories and inject into the system prompt
    3. Get the agent's response
    """
    # Step 1: extract and store any new personal facts
    new_facts = extract_facts(user_input)
    for fact in new_facts:
        add_memory(fact)

    # Step 2: refresh the agent's system prompt with all memories
    all_facts = get_all_user_memories()
    agent.system_prompt = SYSTEM_PROMPT.format(
        memory_context=format_memories_for_prompt(all_facts)
    )

    # Step 3: get the reply
    response = agent(user_input)
    return str(response)


# ================================================================
# SECTION 5 — Interactive chat loop
# ================================================================

if __name__ == "__main__":
    print("=" * 58)
    print("  Challenge 3 — Memory Agent (Strands + Ollama)")
    print("  Memory saved to: memory_store/memories.json")
    print("  Persists across restarts.")
    print("=" * 58)
    print("\nTry saying:")
    print("  • My name is Thilaga")
    print("  • I love Python programming")
    print("  • I am from Chennai")
    print("  • What is my name?")
    print("  • What do I love?")
    print("\nSpecial commands:")
    print("  memories → show all stored memories")
    print("  clear    → delete all memories")
    print("  exit     → quit")
    print("-" * 58 + "\n")

    while True:
        user_input = input("You: ").strip()

        # Exit
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye! Your memories are saved.")
            break

        # Skip blank input
        if not user_input:
            continue

        # Show all stored memories
        if user_input.lower() == "memories":
            facts = get_all_user_memories()
            if facts:
                print("\n📋 Stored Memories:")
                for i, f in enumerate(facts, 1):
                    print(f"  {i}. {f}")
                print()
            else:
                print("\n  No memories stored yet.\n")
            continue

        # Clear all memories
        if user_input.lower() == "clear":
            clear_user_memories()
            print("\n  All memories cleared.\n")
            continue

        # Normal chat
        response = chat(user_input)
        print(f"\nAgent: {response}\n")
        print("-" * 58)





