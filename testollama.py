import os
from typing import List, Optional, Dict, Any
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Define the state structure used by LangGraph
class State(TypedDict):
    messages: List[BaseMessage]

class OllamaAgent:
    """
    A simple LangGraph agent that uses a local Ollama model.
    """
    def __init__(self, model: str = "gemma3:1b", base_url: str = "http://localhost:11434"):
        """
        Args:
            model: Name of the model pulled in Ollama (e.g., "gemma3:1b", "mistral").
            base_url: Ollama server URL (default is localhost:11434).
        """
        self.model = model
        # Create the Ollama LLM client
        self.llm = ChatOllama(
            model=self.model,
            base_url=base_url,      # optional, defaults to http://localhost:11434
            temperature=0.7
        )
        # Build the graph once during initialization
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build and compile the LangGraph workflow."""
        builder = StateGraph(State)
        builder.add_node("call_model", self._call_model)
        builder.set_entry_point("call_model")
        builder.add_edge("call_model", END)
        return builder.compile()

    def _call_model(self, state: State) -> Dict[str, List[BaseMessage]]:
        """Node function: call the LLM and return updated messages."""
        messages = state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": messages + [response]}

    def run(self, user_input: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """
        Non‑streaming invocation.
        Returns the assistant's final reply as a string.
        """
        initial_state = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ]
        }
        final_state = self.graph.invoke(initial_state)
        return final_state["messages"][-1].content

    def stream(self, user_input: str, system_prompt: str = "You are a helpful assistant."):
        """
        Streaming generator that yields chunks of the response as they arrive.
        """
        initial_state = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ]
        }
        for event in self.graph.stream(initial_state):
            for node_name, node_output in event.items():
                if "messages" in node_output:
                    new_msg = node_output["messages"][-1]
                    if hasattr(new_msg, "content"):
                        yield new_msg.content