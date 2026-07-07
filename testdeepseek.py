import os
from typing import List, Optional, Dict, Any
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Define the state structure used by LangGraph
class State(TypedDict):
    messages: List[BaseMessage]

class DeepSeekAgent:
    """
    A simple LangGraph agent that uses DeepSeek's API.
    """
    def __init__(self, model: str = "deepseek-v4-pro", api_key: Optional[str] = None):
        # Use provided key, else read from environment
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key not found. Set DEEPSEEK_API_KEY env var or pass it.")
        
        self.model = model
        self.llm = ChatOpenAI(
            model=self.model,
            openai_api_key=self.api_key,
            base_url="https://api.deepseek.com"
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
        # The last message is the AI's response
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