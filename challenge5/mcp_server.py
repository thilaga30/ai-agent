# challenge5/mcp_server.py
# ============================================================
# Local MCP Server
#
# This is a Model Context Protocol (MCP) server that exposes
# tools over stdio. The Strands agent launches this as a
# subprocess and communicates via stdin/stdout using the MCP
# protocol (JSON-RPC under the hood).
#
# Setup:
#   pip install mcp
#
# You do NOT run this file directly — starter.py launches it.
# ============================================================

from datetime import date, datetime
import math
from mcp.server.fastmcp import FastMCP

# Create the MCP server instance with a name
# FastMCP handles all the protocol boilerplate automatically
mcp = FastMCP("local-tools-server")


# ─────────────────────────────────────────────
# MCP TOOL 1: Calculator
# Exposed to the agent as a callable MCP tool
# ─────────────────────────────────────────────
@mcp.tool()
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Supports basic arithmetic (+, -, *, /), power (**), and
    math functions like sqrt, sin, cos, pi, e.

    Args:
        expression: Math expression string, e.g. 'sqrt(144)' or '2 ** 10'
    """
    try:
        # Provide safe math context — no dangerous builtins
        safe_globals = {
            "__builtins__": {},
            "sqrt": math.sqrt,
            "sin":  math.sin,
            "cos":  math.cos,
            "tan":  math.tan,
            "log":  math.log,
            "pi":   math.pi,
            "e":    math.e,
            "abs":  abs,
            "round": round,
        }
        result = eval(expression, safe_globals)
        return f"Result: {result}"
    except Exception as ex:
        return f"Error: {ex}"


# ─────────────────────────────────────────────
# MCP TOOL 2: Get current date and time
# ─────────────────────────────────────────────
@mcp.tool()
def get_datetime() -> str:
    """
    Return the current date and time.
    Use this when the user asks what time or date it is.
    """
    now = datetime.now()
    return (
        f"Current date: {now.strftime('%A, %B %d, %Y')}\n"
        f"Current time: {now.strftime('%I:%M %p')}"
    )


# ─────────────────────────────────────────────
# MCP TOOL 3: Weather lookup (simulated)
# ─────────────────────────────────────────────
@mcp.tool()
def get_weather(city: str) -> str:
    """
    Get simulated weather information for a city.
    No API key required — returns local demo data.

    Args:
        city: Name of the city, e.g. 'Tokyo'
    """
    weather_db = {
        "london":    {"temp": "15°C", "condition": "Cloudy",        "humidity": "78%"},
        "new york":  {"temp": "22°C", "condition": "Sunny",         "humidity": "55%"},
        "tokyo":     {"temp": "28°C", "condition": "Humid",         "humidity": "82%"},
        "paris":     {"temp": "17°C", "condition": "Rainy",         "humidity": "85%"},
        "dubai":     {"temp": "38°C", "condition": "Hot",           "humidity": "40%"},
        "chennai":   {"temp": "34°C", "condition": "Hot & Humid",   "humidity": "80%"},
        "bangalore": {"temp": "26°C", "condition": "Partly Cloudy", "humidity": "65%"},
        "mumbai":    {"temp": "32°C", "condition": "Humid",         "humidity": "88%"},
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
        "Try: London, New York, Tokyo, Paris, Dubai, Chennai, Bangalore, Mumbai."
    )


# ─────────────────────────────────────────────
# MCP TOOL 4: Age calculator
# ─────────────────────────────────────────────
@mcp.tool()
def calculate_age(birth_year: int, birth_month: int = 1, birth_day: int = 1) -> str:
    """
    Calculate someone's current age from their date of birth.

    Args:
        birth_year:  Year born, e.g. 1995
        birth_month: Month born (1-12), default 1
        birth_day:   Day born (1-31), default 1
    """
    try:
        today = date.today()
        dob   = date(birth_year, birth_month, birth_day)
        if dob > today:
            return "That date is in the future!"
        age = today.year - dob.year - (
            (today.month, today.day) < (dob.month, dob.day)
        )
        try:
            next_bday = date(today.year, dob.month, dob.day)
            if next_bday <= today:
                next_bday = date(today.year + 1, dob.month, dob.day)
        except ValueError:
            next_bday = date(today.year + 1, dob.month, dob.day)
        days_until = (next_bday - today).days
        return (
            f"Age: {age} years old. "
            f"Next birthday in {days_until} day(s) — {next_bday.strftime('%B %d, %Y')}."
        )
    except ValueError as ex:
        return f"Invalid date: {ex}"


# ─────────────────────────────────────────────
# MCP TOOL 5: Unit converter
# ─────────────────────────────────────────────
@mcp.tool()
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert a value between common units.
    Supported conversions:
      Temperature : celsius↔fahrenheit↔kelvin
      Length      : km↔miles, meters↔feet
      Weight      : kg↔pounds

    Args:
        value:     The numeric value to convert
        from_unit: Source unit (e.g. 'celsius')
        to_unit:   Target unit (e.g. 'fahrenheit')
    """
    f = from_unit.lower().strip()
    t = to_unit.lower().strip()

    conversions = {
        ("celsius",    "fahrenheit"): lambda v: v * 9/5 + 32,
        ("fahrenheit", "celsius"):    lambda v: (v - 32) * 5/9,
        ("celsius",    "kelvin"):     lambda v: v + 273.15,
        ("kelvin",     "celsius"):    lambda v: v - 273.15,
        ("km",         "miles"):      lambda v: v * 0.621371,
        ("miles",      "km"):         lambda v: v * 1.60934,
        ("meters",     "feet"):       lambda v: v * 3.28084,
        ("feet",       "meters"):     lambda v: v / 3.28084,
        ("kg",         "pounds"):     lambda v: v * 2.20462,
        ("pounds",     "kg"):         lambda v: v / 2.20462,
    }

    key = (f, t)
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    return (
        f"Conversion from '{from_unit}' to '{to_unit}' not supported. "
        "Supported: celsius/fahrenheit/kelvin, km/miles, meters/feet, kg/pounds."
    )


# ─────────────────────────────────────────────
# Entry point — run the MCP server over stdio
# starter.py launches this via subprocess
# ─────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="stdio")
