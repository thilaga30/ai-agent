# challenge2/starter.py
# Tools Agent — Strands SDK + Ollama llama3.2:3b
# Tools: calculator, weather, age calculator

from datetime import date
from strands import Agent, tool
from strands.models.ollama import OllamaModel

# ─────────────────────────────────────────────
# TOOL 1: Calculator
# ─────────────────────────────────────────────
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
        # eval is scoped to only allow math — no builtins exposed
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


# ─────────────────────────────────────────────
# TOOL 2: Weather (simulated)
# ─────────────────────────────────────────────
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
    # Simulated data — swap this dict for a real API call if you want live data
    weather_db = {
        "london":    {"temp": "15°C", "condition": "Cloudy",  "humidity": "78%"},
        "new york":  {"temp": "22°C", "condition": "Sunny",   "humidity": "55%"},
        "tokyo":     {"temp": "28°C", "condition": "Humid",   "humidity": "82%"},
        "sydney":    {"temp": "18°C", "condition": "Windy",   "humidity": "60%"},
        "paris":     {"temp": "17°C", "condition": "Rainy",   "humidity": "85%"},
        "dubai":     {"temp": "38°C", "condition": "Hot",     "humidity": "40%"},
        "bangalore": {"temp": "26°C", "condition": "Partly Cloudy", "humidity": "65%"},
        "mumbai":    {"temp": "32°C", "condition": "Humid",   "humidity": "88%"},
    }

    key = city.lower().strip()
    if key in weather_db:
        w = weather_db[key]
        return (
            f"Weather in {city.title()}: "
            f"{w['condition']}, {w['temp']}, Humidity: {w['humidity']}"
        )
    return (
        f"Weather data not available for '{city}'. "
        f"Try: London, New York, Tokyo, Sydney, Paris, Dubai, Bangalore, Mumbai."
    )


# ─────────────────────────────────────────────
# TOOL 3: Age Calculator
# ─────────────────────────────────────────────
@tool
def calculate_age(birth_year: int, birth_month: int = 1, birth_day: int = 1) -> str:
    """
    Calculate a person's age from their date of birth.

    Args:
        birth_year:  The year they were born, e.g. 1995
        birth_month: The month they were born (1–12), default 1
        birth_day:   The day they were born (1–31), default 1

    Returns:
        Their current age and next birthday info.
    """
    try:
        today = date.today()
        dob = date(birth_year, birth_month, birth_day)

        if dob > today:
            return "That birth date is in the future!"

        # Calculate full years elapsed
        age = today.year - dob.year - (
            (today.month, today.day) < (dob.month, dob.day)
        )

        # Next birthday
        try:
            next_birthday = date(today.year, dob.month, dob.day)
            if next_birthday <= today:
                next_birthday = date(today.year + 1, dob.month, dob.day)
            days_until = (next_birthday - today).days
        except ValueError:
            # Edge case: Feb 29 in a non-leap year
            next_birthday = date(today.year + 1, dob.month, dob.day)
            days_until = (next_birthday - today).days

        return (
            f"Age: {age} years old. "
            f"Next birthday in {days_until} day(s) on {next_birthday.strftime('%B %d, %Y')}."
        )
    except ValueError as e:
        return f"Invalid date: {e}"


# ─────────────────────────────────────────────
# MODEL + AGENT SETUP
# ─────────────────────────────────────────────
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.7,
)

agent = Agent(
    model=ollama_model,
    system_prompt=(
        "You are a helpful assistant with access to three tools:\n"
        "1. calculator — for any math expressions\n"
        "2. get_weather — for current weather in a city\n"
        "3. calculate_age — to find someone's age from their birth date\n\n"
        "Always use the appropriate tool when the user's question calls for it. "
        "Keep answers short and clear."
    ),
    tools=[calculator, get_weather, calculate_age],
)

# ─────────────────────────────────────────────
# INTERACTIVE CHAT LOOP
# ─────────────────────────────────────────────
print("=" * 50)
print("  Tools Agent — Challenge 2")
print("=" * 50)
print("Ask me anything! Try:")
print("  • 'What is (25 * 4) + 100?'")
print("  • 'What is the weather in Tokyo?'")
print("  • 'How old is someone born in 1995?'")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ("quit", "exit", "q"):
        print("GoodBye!")
        break

    if not user_input:
        continue

    response = agent(user_input)
    print(f"Agent: {response}\n")
