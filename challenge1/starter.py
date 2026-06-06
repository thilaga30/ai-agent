# challenge1/starter.py
# A simple AI agent using Strands SDK with a local Ollama model (llama3.2:3b)

# Step 1: Import the Agent class from the strands library
from strands import Agent

# Step 2: Import OllamaModel so we can point the agent at our local Ollama server
from strands.models.ollama import OllamaModel

# Step 3: Create the model instance
# - host: the address where Ollama is running (default port is 11434)
# - model_id: the model we pulled with `ollama pull llama3.2:3b`
# - temperature: controls creativity (0.0 = focused/deterministic, 1.0 = creative/random)
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.7,
)

# Step 4: Create the agent
# - model: the LLM backend the agent will use
# - system_prompt: gives the agent its personality / role
agent = Agent(
    model=ollama_model,
    system_prompt="You are a helpful assistant. Keep your answers short and clear.",
)

# Step 5: Run a simple conversation loop so we can chat with the agent
print("Agent ready. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()

    # Exit condition
    if user_input.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break

    # Skip empty input
    if not user_input:
        continue

    # Step 6: Send the user message to the agent and print the response
    # Calling agent(...) returns a response object; str() gives us the text
    response = agent(user_input)
    print(f"Agent: {response}\n")
