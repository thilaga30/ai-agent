# challenge4/starter.py
# ============================================================
# Full Agent — Strands SDK + Ollama llama3.2:3b
# Combines: Calculator + Weather + Age Calculator + Memory
#
# Setup (run once):
#   pip install "strands-agents[ollama]"
#   ollama pull llama3.2:3b
# ============================================================

import os
import json
import logging
from datetime import date
from strands import Agent, tool
from strands.models.ollama import OllamaModel

# Suppress noisy telemetry warnings from ChromaDB / PostHog
logging.getLogger("backoff").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)


# ============================================================
# SECTION 1: Persistent Memory (JSON file on disk)
#
# Memories are stored as a plain JSON file so they survive
# restarts. No external database or API key needed.
# Format: { "user_id": ["fact1", "fact2", ...] }
# ============================================================

MEMORY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "memory_store",
    "memories.json"
)

USER_ID = "user"   # change this to scope memories per person


def load_memories() -> dict:
    """Load all memories from disk. Returns {} if file doesn't exist."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}


def save_memories(memories: dict) -> None:
    """Persist memories to disk, creating the folder if needed."""
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump(memories, f, indent=2)


def add_memory(fact: str) -> None:
    """Add a single fact for the current user, avoiding duplicates."""
    memories = load_memories()
    user_facts = memories.get(USER_ID, [])
    if fact not in user_facts:
        user_facts.append(fact)
    memories[USER_ID] = user_facts
    save_memories(memories)


def get_user_memories() -> list:
    """Return all stored facts for the current user."""
    return load_memories().get(USER_ID, [])


def clear_user_memories() -> None:
    """Wipe all stored facts for the current user."""
    memories = load_memories()
    memories[USER_ID] = []
    save_memories(memories)


def format_memory_block(facts: list) -> str:
    """Format facts as a readable string to inject into the system prompt."""
    if not facts:
        return "Nothing stored yet."
    return "\n".join(f"  - {f}" for f in facts)


# ============================================================
# SECTION 2: Fact Extractor (lightweight LLM pass)
#
# Before replying, we run the user's message through a tiny
# extractor agent that pulls out any personal facts and stores
# them. This is separate from the main agent so their contexts
# don't interfere with each other.
# ============================================================

_extractor_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
)

_extractor = Agent(
    model=_extractor_model,
    system_prompt=(
        "You are a fact extractor. "
        "Given a message, extract any personal facts the user is sharing about themselves. "
        "Return ONLY a JSON array of short fact strings. "
        "Example input:  'My name is Thilag and I love Python' "
        "Example output: [\"Name is Thilag\", \"Loves Python\"] "
        "If there are no personal facts, return: [] "
        "Return ONLY the JSON array, nothing else."
    ),
)


def extract_and_store_facts(user_message: str) -> None:
    """Run the extractor on user input and save any new facts to memory."""
    try:
        raw = str(_extractor(user_message)).strip()
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return
        facts = json.loads(raw[start:end])
        for fact in facts:
            if isinstance(fact, str) and fact.strip():
                add_memory(fact)
    except Exception:
        pass  # silently skip if extraction fails


# ============================================================
# SECTION 3: Tools
#
# Each function decorated with @tool becomes a callable skill
# for the agent. The docstring is what the LLM reads to decide
# when and how to use the tool — write it clearly.
# ============================================================

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a basic math expression and return the result.
    Supports +, -, *, /, ** (power), and parentheses.

    Args:
        expression: A math expression string, e.g. '(10 + 5) * 2'

    Returns:
        The result as a string, or an error message.
    """
    try:
        # Sandboxed eval — no builtins exposed, math only
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a given city.
    Returns simulated weather data (no API key needed).

    Args:
        city: Name of the city, e.g. 'London'

    Returns:
        A weather summary string.
    """
    weather_db = {
        "london":    {"temp": "15°C", "condition": "Cloudy",        "humidity": "78%"},
        "new york":  {"temp": "22°C", "condition": "Sunny",         "humidity": "55%"},
        "tokyo":     {"temp": "28°C", "condition": "Humid",         "humidity": "82%"},
        "sydney":    {"temp": "18°C", "condition": "Windy",         "humidity": "60%"},
        "paris":     {"temp": "17°C", "condition": "Rainy",         "humidity": "85%"},
        "dubai":     {"temp": "38°C", "condition": "Hot",           "humidity": "40%"},
        "bangalore": {"temp": "26°C", "condition": "Partly Cloudy", "humidity": "65%"},
        "mumbai":    {"temp": "32°C", "condition": "Humid",         "humidity": "88%"},
        "chennai":   {"temp": "34°C", "condition": "Hot & Humid",   "humidity": "80%"},
    }
    key = city.lower().strip()
    if key in weather_db:
        w = weather_db[key]
        return (
            f"Weather in {city.title()}: "
            f"{w['condition']}, {w['temp']}, Humidity: {w['humidity']}"
        )
    return (
        f"No data for '{city}'. "
        "Available: London, New York, Tokyo, Sydney, Paris, Dubai, Bangalore, Mumbai, Chennai."
    )


@tool
def calculate_age(birth_year: int, birth_month: int = 1, birth_day: int = 1) -> str:
    """
    Calculate a person's age from their date of birth.

    Args:
        birth_year:  Year of birth, e.g. 1995
        birth_month: Month of birth (1–12), default 1
        birth_day:   Day of birth (1–31), default 1

    Returns:
        Current age and days until next birthday.
    """
    try:
        today = date.today()
        dob   = date(birth_year, birth_month, birth_day)

        if dob > today:
            return "That birth date is in the future!"

        age = today.year - dob.year - (
            (today.month, today.day) < (dob.month, dob.day)
        )

        try:
            next_bday = date(today.year, dob.month, dob.day)
            if next_bday <= today:
                next_bday = date(today.year + 1, dob.month, dob.day)
        except ValueError:  # Feb 29 in non-leap year
            next_bday = date(today.year + 1, dob.month, dob.day)

        days_until = (next_bday - today).days
        return (
            f"Age: {age} years old. "
            f"Next birthday in {days_until} day(s) on {next_bday.strftime('%B %d, %Y')}."
        )
    except ValueError as e:
        return f"Invalid date: {e}"


# ============================================================
# SECTION 4: Main Agent Setup
#
# The system prompt is rebuilt before every message so it always
# contains the latest memories. The three tools are registered
# so the agent can call them autonomously when needed.
# ============================================================

SYSTEM_PROMPT_TEMPLATE = """You are a helpful personal assistant with memory and useful tools.

What you remember about the user:
--- MEMORY ---
{memory_block}
--------------

Tools available to you:
  • calculator    — evaluate math expressions
  • get_weather   — current weather for a city
  • calculate_age — calculate age from a birth date

Always use a tool when the user's question calls for it.
Use the memory to personalise responses — greet the user by name if you know it.
Be friendly and concise."""

_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.7,
)

agent = Agent(
    model=_model,
    system_prompt=SYSTEM_PROMPT_TEMPLATE.format(
        memory_block=format_memory_block(get_user_memories())
    ),
    tools=[calculator, get_weather, calculate_age],
)


# ============================================================
# SECTION 5: Chat function
#
# Each turn:
#   1. Extract + store any personal facts from user input
#   2. Refresh the agent's system prompt with all memories
#   3. Call the agent — it decides whether to use a tool or reply directly
# ============================================================

def chat(user_input: str) -> str:
    # Step 1 — extract and persist any new facts
    extract_and_store_facts(user_input)

    # Step 2 — rebuild system prompt with the latest memories
    agent.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        memory_block=format_memory_block(get_user_memories())
    )

    # Step 3 — get the response (agent may call a tool internally)
    return str(agent(user_input))


# ============================================================
# SECTION 6: Interactive Chat Loop
# ============================================================

if __name__ == "__main__":
    print("=" * 58)
    print("  Challenge 4 — Full Agent (Tools + Memory)")
    print("  Memories saved to: memory_store/memories.json")
    print("=" * 58)
    print("\nTry asking:")
    print("  • My name is Thilag and I'm from Chennai")
    print("  • What is the weather in Chennai?")
    print("  • What is (125 * 8) + 200?")
    print("  • How old is someone born on June 10, 1998?")
    print("  • What is my name?  ← tests memory recall")
    print("\nSpecial commands:")
    print("  memories → show all stored memories")
    print("  clear    → delete all memories")
    print("  quit     → exit")
    print("-" * 58 + "\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye! Memories are saved for next time.")
            break

        # Show stored memories
        if user_input.lower() == "memories":
            facts = get_user_memories()
            if facts:
                print("\n[Stored Memories]")
                for i, f in enumerate(facts, 1):
                    print(f"  {i}. {f}")
                print()
            else:
                print("\n  No memories stored yet.\n")
            continue

        # Clear memories
        if user_input.lower() == "clear":
            clear_user_memories()
            print("\n  Memories cleared.\n")
            continue

        response = chat(user_input)
        print(f"\nAgent: {response}\n")
        print("-" * 58)
