"""Enhanced Microsoft Agent Framework retail operations demo with DevUI and observability.

================================================================================
MICROSOFT AGENT FRAMEWORK - ARCHITECTURE OVERVIEW
================================================================================

This demo showcases the key concepts and patterns of the Microsoft Agent Framework:

1. CORE BUILDING BLOCKS:
   - ChatAgent: AI agents with specialized roles, instructions, and tools
   - AzureOpenAIChatClient: LLM client for Azure OpenAI Service integration
   - WorkflowBuilder: Orchestrator for multi-agent coordination via directed graphs
   - Tools: Python functions that agents can call (with automatic schema generation)

2. AGENT LIFECYCLE:
   a. Configuration: Define name, instructions (system prompt), and tools
   b. Instantiation: Create ChatAgent with bound AzureOpenAIChatClient
   c. Execution: Call agent.run(message) to process user requests
   d. Tool Calling: Framework handles LLM → tool → LLM roundtrips automatically
   e. Response: Agent returns structured result with text and execution metadata

3. MULTI-AGENT WORKFLOWS:
   - WorkflowBuilder creates directed graphs where nodes = agents, edges = handoffs
   - set_start_executor() designates the entry point agent
   - add_edge(agent_A, agent_B) enables agent_A to hand control to agent_B
   - Workflow execution produces EventCollection with all agent outputs
   - Context automatically flows between agents during handoffs

4. TOOL INTEGRATION:
   - Tools are standard Python functions with Annotated type hints
   - Pydantic Field() provides semantic descriptions for the LLM
   - Framework auto-generates OpenAI function schemas from signatures
   - No manual schema writing or JSON serialization needed

5. OBSERVABILITY & DEBUGGING:
   - OpenTelemetry integration for distributed tracing
   - AgentRunEvent objects for real-time monitoring
   - DevUI for interactive testing and visualization
   - Full execution traces with LLM requests, tool calls, and responses

6. AZURE INTEGRATION:
   - AzureOpenAIChatClient handles authentication and deployment connection
   - Supports both API key (AzureKeyCredential) and AAD (AzureCliCredential)
   - Environment variables for configuration (AZURE_OPENAI_ENDPOINT, etc.)

Set these variables in `.env`:
* `AZURE_OPENAI_ENDPOINT` – Azure OpenAI resource endpoint
* `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME` – Chat deployment name (e.g., gpt-4o-mini)
* `AZURE_OPENAI_API_KEY` or authenticate with `az login` for AAD auth
* Optional: `AZURE_OPENAI_API_VERSION` to pin the API version

Usage:
* Run CLI demos: `python maf_enhanced.py`
* Launch DevUI: `python maf_enhanced.py --devui`
* Custom DevUI port: `python maf_enhanced.py --devui --devui-port 8080`
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import os
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Final

from dotenv import load_dotenv

# ============================================================================
# MICROSOFT AGENT FRAMEWORK - CORE IMPORTS
# ============================================================================
# The Microsoft Agent Framework provides three primary building blocks:
#
# 1. ChatAgent: The core abstraction representing an AI agent with:
#    - name: Unique identifier for the agent
#    - instructions: System prompt defining the agent's role and behavior
#    - tools: Functions the agent can call to interact with external systems
#    - chat_client: The underlying LLM client (e.g., AzureOpenAI)
#
# 2. WorkflowBuilder: A builder pattern for orchestrating multiple agents:
#    - set_start_executor(): Designates which agent begins the workflow
#    - add_edge(): Defines handoffs between agents (directed graph edges)
#    - build(): Compiles the workflow into an executable graph
#    This enables complex multi-agent coordination where agents can pass
#    control to one another based on task requirements.
#
# 3. AgentRunEvent: Event objects emitted during agent/workflow execution:
#    - executor_id: Which agent produced this event
#    - data: The content/message from the agent
#    - Enables real-time monitoring of agent activities during execution
#
from agent_framework import ChatAgent, WorkflowBuilder, AgentRunEvent

# ============================================================================
# MICROSOFT AGENT FRAMEWORK - AZURE INTEGRATION
# ============================================================================
# AzureOpenAIChatClient: Azure-specific LLM client that:
# - Handles authentication via Azure credentials (API key or AAD)
# - Connects to Azure OpenAI Service deployments
# - Provides the model interface that ChatAgent uses for inference
# - Supports both direct API key auth and Azure CLI credential auth
#
from agent_framework.azure import AzureOpenAIChatClient

# Azure authentication primitives:
# - AzureKeyCredential: Direct API key authentication
# - AzureCliCredential: Uses 'az login' session for AAD authentication
from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureCliCredential

# Pydantic Field for annotating function parameters with descriptions.
# The Agent Framework uses these annotations to generate tool schemas
# that the LLM can understand and invoke correctly.
from pydantic import Field

# Import WorkflowViz for visualization (optional, graceful fallback)
try:
    from agent_framework import WorkflowViz
    WORKFLOW_VIZ_AVAILABLE = True
except ImportError:
    WORKFLOW_VIZ_AVAILABLE = False


logger = logging.getLogger(__name__)


# ANSI color codes for terminal output
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
FG_BLUE = "\033[34m"
FG_CYAN = "\033[36m"
FG_GREEN = "\033[32m"
FG_MAGENTA = "\033[35m"
FG_RED = "\033[31m"
FG_YELLOW = "\033[33m"
FG_GRAY = "\033[90m"


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.getenv("TERM", "") not in {"", "dumb"}


_USE_COLOR: Final[bool] = _supports_color()


def _colorize(text: str, *codes: str) -> str:
    if not text or not _USE_COLOR:
        return text
    return "".join(codes) + text + RESET


def _banner(title: str, width: int = 70) -> None:
    width = max(width, len(title))
    line = _colorize("=" * width, DIM)
    print(line)
    print(_colorize(title, BOLD, FG_MAGENTA))
    print(line)


def _heading(text: str) -> str:
    return _colorize(text, BOLD, FG_CYAN)


def _label(text: str) -> str:
    return _colorize(text, FG_BLUE)


def _success(text: str) -> str:
    return _colorize(text, FG_GREEN)


def _warning(text: str) -> str:
    return _colorize(text, BOLD, FG_YELLOW)


def _muted(text: str) -> str:
    return _colorize(text, FG_GRAY)


def _kv(label: str, value: str, indent: str = "   ") -> str:
    return f"{indent}{_label(label)}: {value}"


# Scenario prompts
DIRECT_PROMPT: Final[str] = (
    "Call get_store_snapshot for Store 108 and summarize urgent actions for the district lead."
)
AGENT_PROMPT: Final[str] = (
    "You just received the latest sensor feed for Store 108. Draft a floor stack update for the store crew."
)
STREAM_PROMPT: Final[str] = (
    "Report the next three real-time alerts as they occur. Use short imperative sentences."
)
WORKFLOW_PROMPT: Final[str] = (
    "Prepare a joint plan for Store 108 covering restock timing, labor shifts, and promotion adjustments."
)
HUMAN_LOOP_PROMPT: Final[str] = (
    "Store 108 has low conversion rates. Analyze the data and propose a promotional discount if needed."
)
DEVUI_AGENT_NAME: Final[str] = "RetailOpsAgent"
DEVUI_AGENT_DESCRIPTION: Final[str] = (
    "Coordinates flagship retail operations updates using live telemetry and tools."
)
DEVUI_AGENT_INSTRUCTIONS: Final[str] = (
    "You coordinate operations for Store 108. Call the `get_store_snapshot` tool before you respond so you "
    "always reference current telemetry. Summarize critical merchandising, staffing, and supply actions."
)
WORKFLOW_NAME: Final[str] = "RetailOpsWorkflow"


def _mask(value: str | None) -> str:
    if not value:
        return "<not set>"
    stripped = value.strip()
    if len(stripped) <= 6:
        return "*" * len(stripped)
    return f"{stripped[:3]}...{stripped[-3:]}"


def _print_version_details() -> None:
    import agent_framework

    package_path = getattr(agent_framework, "__file__", "<unknown>")
    print(_heading("🧾 Agent Framework core"))
    print(_kv("version", _muted(getattr(agent_framework, "__version__", "unknown"))))
    print(_kv("location", _muted(package_path)))

    try:
        azure_pkg = importlib.import_module("agent_framework.azure")
    except ModuleNotFoundError:
        print(_warning("❌ agent_framework.azure module not found"))
    else:
        azure_version = getattr(azure_pkg, "__version__", "bundled with core")
        azure_path = getattr(azure_pkg, "__file__", "<unknown>")
        print(_heading("🧾 Agent Framework Azure helpers"))
        print(_kv("version", _muted(azure_version)))
        print(_kv("location", _muted(azure_path)))

    try:
        obs_pkg = importlib.import_module("agent_framework.observability")
        print(_heading("🧾 Agent Framework Observability"))
        print(_kv("status", _success("Available")))
    except ModuleNotFoundError:
        print(_warning("❌ agent_framework.observability module not found"))

    try:
        devui_pkg = importlib.import_module("agent_framework.devui")
        print(_heading("🧾 Agent Framework DevUI"))
        print(_kv("status", _success("Available")))
    except ModuleNotFoundError:
        print(_warning("❌ agent_framework.devui module not found (optional)"))
        print(_muted("   Install with: pip install agent-framework-devui --pre"))


def _ensure_azure_openai_env() -> None:
    required = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"]
    missing = [name for name in required if not os.getenv(name)]
    print(_heading("🌐 Azure OpenAI configuration"))
    for name in required:
        value = _mask(os.getenv(name))
        styled_value = _warning(value) if name in missing else _success(value)
        print(_kv(name, styled_value))

    # Show optional API version
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    if api_version:
        print(_kv("AZURE_OPENAI_API_VERSION", _success(api_version)))

    if missing:
        joined = ", ".join(missing)
        print(_warning(f"⚠ Missing required environment variables: {joined}"))
        raise RuntimeError(f"Missing required environment variables: {joined}")


def _choose_credential() -> AzureKeyCredential | AzureCliCredential:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if api_key:
        print(_success("🔑 Using Azure OpenAI API key authentication."))
        return AzureKeyCredential(api_key)

    print(_warning("🔐 No API key detected – falling back to Azure CLI authentication."))
    print(_muted("   Ensure `az login` succeeded in this shell."))
    return AzureCliCredential()


def _simulate_store_metrics(store_id: str) -> dict[str, str]:
    """Create a deterministic but time-sensitive snapshot for a store."""
    now = datetime.now(UTC)
    timestamp = now.strftime("%Y-%m-%d %H:%M UTC")
    seed = f"{store_id}-{now:%Y%m%d%H%M}"
    rng = random.Random(seed)

    foot_traffic = rng.randint(240, 420)
    conversion_rate = round(rng.uniform(0.16, 0.28), 3)
    net_sales = rng.randint(52000, 86000)
    curbside_orders = rng.randint(18, 42)
    on_hand_units = rng.randint(1800, 2600)
    backroom_units = rng.randint(320, 680)
    labor_gap = rng.choice(["0 hrs", "2 hrs", "4 hrs", "6 hrs"])

    top_oos_skus = [
        f"SKU-{rng.randint(1000, 1999)} ({rng.choice(['sneakers', 'hoodies', 'smartwatches', 'duffle bags'])})",
        f"SKU-{rng.randint(2000, 2999)} ({rng.choice(['caps', 'leggings', 'wireless buds', 'trail shoes'])})",
    ]

    upcoming_events = [
        "Flash promo @ 4pm needs front table reset",
        "Mall sports night driving late traffic uptick",
        "New loyalty kiosk going live tomorrow",
    ]

    return {
        "timestamp": timestamp,
        "store_id": store_id,
        "foot_traffic": str(foot_traffic),
        "conversion_rate": f"{conversion_rate * 100:.1f}%",
        "net_sales": f"${net_sales:,}",
        "curbside_orders": str(curbside_orders),
        "on_hand_units": str(on_hand_units),
        "backroom_units": str(backroom_units),
        "labor_gap": labor_gap,
        "top_oos": ", ".join(top_oos_skus),
        "events": "; ".join(upcoming_events),
    }


# ============================================================================
# AGENT TOOL DEFINITION
# ============================================================================
# This function demonstrates how to create tools that agents can call.
# Key aspects of Microsoft Agent Framework tool design:
#
# 1. Type Annotations: The function signature uses Annotated[type, Field(...)]
#    to provide both type information AND semantic descriptions that the LLM
#    uses to understand when and how to call the tool.
#
# 2. Pydantic Field: Field(description="...") creates metadata that becomes
#    part of the tool schema sent to the LLM. The LLM reads this description
#    to decide if this tool is relevant for the user's request.
#
# 3. Return Type: The function returns a string that the agent will receive
#    and incorporate into its reasoning/response. More complex tools can
#    return structured data (dicts, dataclasses, Pydantic models).
#
# 4. Automatic Schema Generation: The Agent Framework automatically converts
#    this Python function into an OpenAI function-calling schema that the
#    LLM understands, handling all serialization/deserialization.
#
def get_store_snapshot(
    store_id: Annotated[str, Field(description="Store identifier to inspect.")],
) -> str:
    """Return a formatted operations snapshot for the requested store."""
    metrics = _simulate_store_metrics(store_id)
    return (
        f"Store {metrics['store_id']} snapshot @ {metrics['timestamp']}\n"
        f"Foot traffic: {metrics['foot_traffic']} | Conversion: {metrics['conversion_rate']}\n"
        f"Net sales: {metrics['net_sales']} | Curbside orders: {metrics['curbside_orders']}\n"
        f"Inventory on hand: {metrics['on_hand_units']} units (backroom {metrics['backroom_units']})\n"
        f"Labor gap: {metrics['labor_gap']}\n"
        f"Top out-of-stock SKUs: {metrics['top_oos']}\n"
        f"Upcoming events: {metrics['events']}"
    )


# ============================================================================
# HUMAN-IN-THE-LOOP TOOL (Requires Approval)
# ============================================================================
# This demonstrates a critical tool that requires human approval before execution.
# In Microsoft Agent Framework, you can implement human-in-the-loop using:
# 1. Request/response workflow patterns
# 2. Tool-level approval gates
# 3. Custom middleware for approval logic
#
def approve_discount(
    store_id: Annotated[str, Field(description="Store identifier for discount application.")],
    discount_percentage: Annotated[int, Field(description="Discount percentage to apply (1-50).")],
    reason: Annotated[str, Field(description="Business reason for the discount.")],
) -> str:
    """Apply a store-wide discount. This is a critical operation requiring human approval.

    In production, this would emit an approval request that routes to a human reviewer.
    For this demo, we simulate the approval process.
    """
    # Simulate human approval request
    print()
    print(_colorize("⚠️  HUMAN APPROVAL REQUIRED", BOLD, FG_YELLOW))
    print("─" * 70)
    print(_label(f"Action: Apply {discount_percentage}% discount to Store {store_id}"))
    print(_label(f"Reason: {reason}"))
    print("─" * 70)

    # Simulate approval prompt
    try:
        user_input = input(_colorize("Approve this action? (yes/no): ", FG_CYAN))
        approved = user_input.strip().lower() in ["yes", "y"]
    except (EOFError, KeyboardInterrupt):
        approved = False
        print()

    print()

    if approved:
        result = f"✅ APPROVED: {discount_percentage}% discount applied to Store {store_id}. Reason: {reason}"
        print(_success(result))
        return result
    else:
        result = f"❌ DENIED: Discount request for Store {store_id} was rejected by human reviewer."
        print(_warning(result))
        return result


@dataclass(slots=True)
class EnhancedContext:
    """Context for enhanced agent framework tests with observability."""
    chat_client: AzureOpenAIChatClient
    model: str


async def run_tool_call_demo(context: EnhancedContext) -> None:
    """Scenario 1: Single agent with tool calling demonstration."""
    print()
    print("=" * 80)
    print(_colorize("🛠️  SCENARIO 1: SINGLE AGENT WITH TOOL CALLING", BOLD, FG_CYAN))
    print("=" * 80)
    print()

    print(_colorize("📋 Overview:", BOLD, FG_YELLOW))
    print("   This scenario demonstrates a single AI agent using tools to fetch real-time data.")
    print("   The agent will call the 'get_store_snapshot' function to retrieve store metrics.")
    print()

    # ========================================================================
    # CREATING AN AGENT - Method 1: Using chat_client.create_agent()
    # ========================================================================
    # This is one of two ways to create agents in the Microsoft Agent Framework.
    #
    # The create_agent() method is a convenience method on AzureOpenAIChatClient
    # that automatically binds the client to the agent. It's equivalent to:
    #   ChatAgent(chat_client=context.chat_client, name=..., ...)
    #
    # Agent constructor parameters:
    # - name: Unique identifier for the agent (used in workflows and events)
    # - instructions: The system prompt that defines the agent's role, behavior,
    #   and decision-making logic. This is where you encode domain expertise.
    # - tools: List of Python functions the agent can call. The framework
    #   automatically generates OpenAI function schemas from these functions.
    #
    # The agent doesn't execute immediately - it's a reusable object that can
    # process multiple requests via agent.run(user_message).
    #
    print(_colorize("🤖 Creating Store Analyst Agent...", BOLD, FG_BLUE))
    agent = context.chat_client.create_agent(
        name="StoreAnalyst",
        instructions="You are a district analyst. Use the get_store_snapshot tool to analyze store data.",
        tools=[get_store_snapshot],
    )
    print(_success("   ✓ Agent created successfully"))
    print(_muted(f"   Name: StoreAnalyst"))
    print(_muted(f"   Tools: get_store_snapshot"))
    print()

    print(_colorize("💬 User Prompt:", BOLD, FG_YELLOW))
    print(_colorize(f'   "{DIRECT_PROMPT}"', FG_GRAY))
    print()

    print(_colorize("⚙️  Agent Processing...", BOLD, FG_MAGENTA))
    print(_muted("   [Step 1] Agent analyzes the request"))
    print(_muted("   [Step 2] Agent decides to call get_store_snapshot tool"))
    print(_muted("   [Step 3] Tool returns store data"))
    print(_muted("   [Step 4] Agent formulates response"))
    print()

    # ========================================================================
    # AGENT EXECUTION - The Run Loop
    # ========================================================================
    # agent.run(message) is the primary execution method for agents.
    #
    # What happens during agent.run():
    # 1. The user message is sent to the LLM along with:
    #    - The agent's system instructions
    #    - The tool schemas (auto-generated from Python functions)
    #    - Conversation history (if any)
    #
    # 2. The LLM responds with either:
    #    - A text response (if no tools needed), OR
    #    - A tool call request (function name + arguments)
    #
    # 3. If tool call requested:
    #    - Framework executes the Python function with LLM-provided arguments
    #    - Tool result is sent back to the LLM
    #    - LLM incorporates result and generates final response
    #
    # 4. The framework returns a result object containing:
    #    - text: The agent's final response
    #    - All intermediate steps and tool calls (for debugging)
    #
    # This is an async operation because it involves network calls to Azure OpenAI.
    #
    result = await agent.run(DIRECT_PROMPT)

    print(_colorize("✨ Agent Response:", BOLD, FG_GREEN))
    print("─" * 80)
    print(_colorize(result.text, FG_GREEN))
    print("─" * 80)
    print()

    print(_success("✅ SCENARIO 1 COMPLETED: Tool calling demonstration successful!"))
    print("=" * 80)
    print()


async def run_human_in_loop_demo(context: EnhancedContext) -> None:
    """Scenario 3: Human-in-the-loop with approval gate demonstration."""
    print()
    print("=" * 80)
    print(_colorize("👤 SCENARIO 3: HUMAN-IN-THE-LOOP WITH APPROVAL", BOLD, FG_CYAN))
    print("=" * 80)
    print()

    print(_colorize("📋 Overview:", BOLD, FG_YELLOW))
    print("   This scenario demonstrates an AI agent that requires human approval for")
    print("   critical operations like applying store-wide discounts.")
    print()

    print(_colorize("🤖 Creating Marketing Agent with Approval Tools...", BOLD, FG_BLUE))

    # Create agent with both regular and approval-required tools
    agent = context.chat_client.create_agent(
        name="MarketingAgent",
        instructions=(
            "You are a marketing analyst. Use get_store_snapshot to analyze store performance. "
            "If conversion rates are below 20%, recommend a promotional discount using approve_discount. "
            "Always explain your reasoning based on the data."
        ),
        tools=[get_store_snapshot, approve_discount],
    )

    print(_success("   ✓ Agent created successfully"))
    print(_muted(f"   Name: MarketingAgent"))
    print(_muted(f"   Tools: get_store_snapshot (auto), approve_discount (requires approval)"))
    print()

    print(_colorize("💬 User Prompt:", BOLD, FG_YELLOW))
    print(_colorize(f'   "{HUMAN_LOOP_PROMPT}"', FG_GRAY))
    print()

    print(_colorize("⚙️  Agent Processing...", BOLD, FG_MAGENTA))
    print(_muted("   [Step 1] Agent will analyze store performance"))
    print(_muted("   [Step 2] Agent may propose a discount"))
    print(_muted("   [Step 3] If discount proposed, human approval will be requested"))
    print(_muted("   [Step 4] Agent will proceed based on approval decision"))
    print()

    result = await agent.run(HUMAN_LOOP_PROMPT)

    print(_colorize("✨ Agent Response:", BOLD, FG_GREEN))
    print("─" * 80)
    print(_colorize(result.text, FG_GREEN))
    print("─" * 80)
    print()

    print(_success("✅ SCENARIO 3 COMPLETED: Human-in-the-loop demonstration successful!"))
    print("=" * 80)
    print()


async def run_multi_agent_demo(context: EnhancedContext, visualize: bool = False, viz_dir: str = "./workflow_visualizations") -> None:
    """Scenario 2: Multi-agent workflow with handoffs demonstration."""
    print()
    print("=" * 80)
    print(_colorize("🔀 SCENARIO 2: MULTI-AGENT WORKFLOW WITH HANDOFFS", BOLD, FG_CYAN))
    print("=" * 80)
    print()

    print(_colorize("📋 Overview:", BOLD, FG_YELLOW))
    print("   This scenario demonstrates multiple AI agents working together in a workflow.")
    print("   Agent 1 (Store Ops) creates a plan, then hands off to Agent 2 (Supply Chain).")
    print()

    print(_colorize("🏗️  Building Multi-Agent Workflow...", BOLD, FG_BLUE))
    print()

    # Create the workflow
    workflow, store_ops_agent, supply_chain_agent = _create_retail_workflow(context.chat_client)

    print(_colorize("   Agent 1: StoreOpsAnalyst", BOLD, FG_BLUE))
    print(_muted("   Role: In-store operations manager"))
    print(_muted("   Tools: get_store_snapshot"))
    print(_muted("   Task: Analyze telemetry and create action plan"))
    print()

    print(_colorize("   Agent 2: SupplyChainLead", BOLD, FG_MAGENTA))
    print(_muted("   Role: Regional supply chain coordinator"))
    print(_muted("   Tools: None (receives handoff)"))
    print(_muted("   Task: Validate plan and add supply chain requirements"))
    print()

    print(_colorize("   Workflow: StoreOpsAnalyst ──handoff──> SupplyChainLead", BOLD, FG_YELLOW))
    print()

    # Generate visualization if requested
    if visualize:
        print(_colorize("📊 Generating Workflow Visualizations...", BOLD, FG_CYAN))
        viz_files = visualize_workflow(workflow, output_dir=viz_dir)
        if viz_files:
            print(_success("   ✓ Workflow diagrams generated!"))
            for fmt, path in viz_files.items():
                print(_muted(f"     • {fmt.upper()}: {path}"))
        print()

    print(_colorize("💬 User Prompt:", BOLD, FG_YELLOW))
    print(_colorize(f'   "{WORKFLOW_PROMPT}"', FG_GRAY))
    print()

    print(_colorize("⚙️  Workflow Execution:", BOLD, FG_MAGENTA))
    print("─" * 80)

    # ========================================================================
    # WORKFLOW EXECUTION - Multi-Agent Coordination
    # ========================================================================
    # workflow.run(message) orchestrates the entire multi-agent workflow.
    #
    # What happens during workflow.run():
    # 1. The message is sent to the start executor (StoreOpsAnalyst)
    # 2. StoreOpsAnalyst processes the message (may call tools)
    # 3. Framework detects if handoff should occur based on workflow edges
    # 4. Context is passed to the next agent (SupplyChainLead)
    # 5. Next agent processes with full context from previous agents
    # 6. Returns an EventCollection containing all events from all agents
    #
    # The result is an EventCollection (events variable) that contains:
    # - All AgentRunEvent objects emitted during execution
    # - Methods to extract outputs and final state
    # - Full execution trace for debugging and observability
    #
    events = await workflow.run(WORKFLOW_PROMPT)

    # ========================================================================
    # PROCESSING WORKFLOW EVENTS - Real-time Monitoring
    # ========================================================================
    # The EventCollection is iterable and contains AgentRunEvent objects.
    #
    # AgentRunEvent properties:
    # - executor_id: The name of the agent that produced this event
    # - data: The message/output from the agent
    # - Additional metadata for debugging and observability
    #
    # This loop demonstrates how to process events in real-time, allowing you to:
    # - Display intermediate results to users
    # - Monitor which agent is currently active
    # - Debug workflow execution
    # - Implement custom logging or observability
    #
    agent_colors = {
        "StoreOpsAnalyst": FG_BLUE,
        "SupplyChainLead": FG_MAGENTA,
    }

    for event in events:
        if isinstance(event, AgentRunEvent):
            agent_name = event.executor_id
            color = agent_colors.get(agent_name, FG_CYAN)
            executor_label = _colorize(f"[{agent_name}]", BOLD, color)
            print(f"{executor_label} {event.data}")

    print("─" * 80)
    print()

    # ========================================================================
    # EXTRACTING WORKFLOW OUTPUTS
    # ========================================================================
    # events.get_outputs() returns the final outputs from all agents.
    # This is useful when you need the final results without processing
    # all intermediate events.
    #
    outputs = events.get_outputs()
    print(_colorize("✨ Final Workflow Outputs:", BOLD, FG_GREEN))
    print("─" * 80)
    for item in outputs:
        print(_colorize(str(item), FG_GREEN))
    print("─" * 80)
    print()

    print(_colorize("📊 Workflow Summary:", BOLD, FG_YELLOW))
    print(_muted(f"   Final State: {events.get_final_state()}"))
    print(_muted(f"   Total Outputs: {len(outputs)}"))
    print()

    print(_success("✅ SCENARIO 2 COMPLETED: Multi-agent workflow with handoffs successful!"))
    print("=" * 80)
    print()




def setup_enhanced_observability() -> None:
    """Configure observability with OpenTelemetry for traces and events."""
    print()
    print(_heading("🔬 Setting up observability"))

    # ========================================================================
    # MICROSOFT AGENT FRAMEWORK - OBSERVABILITY
    # ========================================================================
    # The agent_framework.observability module provides OpenTelemetry integration
    # for comprehensive monitoring and debugging of agent workflows.
    #
    # Key observability features:
    # 1. Distributed Tracing: Tracks the complete execution path across multiple
    #    agents, tool calls, and LLM requests. Each agent run creates a trace
    #    with spans for individual operations.
    #
    # 2. Event Emission: Captures detailed events including:
    #    - Agent starts/completions
    #    - Tool invocations with arguments and results
    #    - LLM requests and responses
    #    - Workflow state transitions
    #
    # 3. OTLP Export: Sends telemetry to OpenTelemetry collectors (e.g.,
    #    Jaeger, Prometheus) for visualization and analysis
    #
    # 4. VS Code Integration: The vs_code_extension_port enables real-time
    #    debugging in VS Code with the Agent Framework extension
    #
    # Configuration options:
    # - enable_otel: Activates OpenTelemetry instrumentation
    # - enable_sensitive_data: Includes LLM prompts/responses in traces
    #   (disable in production for privacy)
    # - otlp_endpoint: Where to send telemetry (local collector or remote)
    # - vs_code_extension_port: Port for VS Code debugging integration
    #
    try:
        from agent_framework.observability import setup_observability

        setup_observability(
            enable_sensitive_data=True,
            otlp_endpoint="http://localhost:4317",
            vs_code_extension_port=4317,
        )
        print(_success("✓ OpenTelemetry observability enabled"))
        print(_kv("OTLP endpoint", "http://localhost:4317"))
        print(_kv("VS Code extension port", "4317"))
        print(_muted("   Traces and events will be captured for debugging"))
    except ImportError:
        print(_warning("⚠ Observability module not available"))
        print(_muted("   Continuing without observability features"))
    except Exception as e:
        print(_warning(f"⚠ Observability setup warning: {e}"))
        print(_muted("   Continuing without observability features"))

    print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhanced retail operations demo for the Microsoft Agent Framework with DevUI support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Demo Scenarios:
  --demo tool-call       Run Scenario 1: Single agent with tool calling
  --demo multi-agent     Run Scenario 2: Multi-agent workflow with handoffs
  --demo human-loop      Run Scenario 3: Human-in-the-loop with approval

Examples:
  python maf_enhanced.py --demo tool-call
  python maf_enhanced.py --demo multi-agent --visualize-workflow
  python maf_enhanced.py --demo human-loop
  python maf_enhanced.py --devui
        """
    )
    parser.add_argument(
        "--demo",
        choices=["tool-call", "multi-agent", "human-loop"],
        help="Run specific demo scenario: 'tool-call', 'multi-agent', or 'human-loop'.",
    )
    parser.add_argument(
        "--devui",
        action="store_true",
        help="Launch DevUI with the retail operations agent and skip CLI demos.",
    )
    parser.add_argument(
        "--devui-port",
        type=int,
        default=8090,
        help="Port to bind the DevUI server (default: 8090).",
    )
    parser.add_argument(
        "--devui-host",
        default="127.0.0.1",
        help="Host to bind the DevUI server (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--devui-no-auto-open",
        action="store_true",
        help="Do not automatically open a browser window when DevUI starts.",
    )
    parser.add_argument(
        "--visualize-workflow",
        action="store_true",
        help="Generate workflow visualization diagrams (Mermaid, SVG, PNG).",
    )
    parser.add_argument(
        "--viz-output-dir",
        default="./workflow_visualizations",
        help="Directory to save workflow visualizations (default: ./workflow_visualizations).",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("MAF_LOG_LEVEL", "INFO"),
        help="Python logging level (default INFO or MAF_LOG_LEVEL env).",
    )
    return parser.parse_args()


def _configure_logging(level_name: str) -> None:
    try:
        numeric_level = int(level_name)
    except (TypeError, ValueError):
        numeric_level = getattr(logging, str(level_name).upper(), None)

    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
        fallback = str(level_name)
    else:
        fallback = None

    logging.basicConfig(level=numeric_level, format="%(message)s")

    if fallback and fallback.upper() != "INFO":
        logger.warning("Unknown log level '%s', defaulted to INFO.", fallback)


def _create_devui_agent(credential: AzureKeyCredential | AzureCliCredential) -> ChatAgent:
    """Create a ChatAgent specifically for DevUI with proper configuration."""

    # ========================================================================
    # MICROSOFT AGENT FRAMEWORK - DEVUI INTEGRATION
    # ========================================================================
    # DevUI is an interactive debugging and testing interface for agents.
    #
    # ChatAgent parameters for DevUI:
    # - name: Unique identifier used in the DevUI interface
    # - description: Human-readable description shown in the UI
    # - instructions: System prompt (same as for CLI/API agents)
    # - chat_client: The LLM client (created inline here)
    # - tools: Available functions the agent can call
    #
    # DevUI provides:
    # - Web-based chat interface for testing agents
    # - Real-time display of tool calls and results
    # - Conversation history and state inspection
    # - Multi-agent workflow visualization
    # - Debug logging and trace viewing
    #
    return ChatAgent(
        name=DEVUI_AGENT_NAME,
        description=DEVUI_AGENT_DESCRIPTION,
        instructions=DEVUI_AGENT_INSTRUCTIONS,
        chat_client=AzureOpenAIChatClient(credential=credential),
        tools=[get_store_snapshot],
    )


def _launch_devui(
    credential: AzureKeyCredential | AzureCliCredential,
    *,
    host: str,
    port: int,
    auto_open: bool,
) -> None:
    """Launch the DevUI server with the retail operations agent."""
    try:
        from agent_framework.devui import serve
    except ImportError as exc:
        raise RuntimeError(
            "Install the DevUI package to launch the debugging UI:\n"
            "  pip install agent-framework-devui --pre"
        ) from exc

    agent = _create_devui_agent(credential)
    logger.info("Starting Retail Operations agent in DevUI")
    logger.info("Available at: http://%s:%s", host, port)
    logger.info("Entity ID: agent_%s", DEVUI_AGENT_NAME)
    print()
    print(_success(f"🚀 DevUI server starting at http://{host}:{port}"))
    print(_muted(f"   Agent Name: {DEVUI_AGENT_NAME}"))
    print(_muted("   Press Ctrl+C to stop the server"))
    print()
    serve(entities=[agent], host=host, port=port, auto_open=auto_open)


def _build_chat_client(credential: AzureKeyCredential | AzureCliCredential) -> AzureOpenAIChatClient:
    """Create a chat client using the configured deployment name."""

    # ========================================================================
    # AZURE OPENAI CHAT CLIENT INITIALIZATION
    # ========================================================================
    # AzureOpenAIChatClient is the foundation for all agent operations.
    #
    # Key responsibilities:
    # 1. Authentication: Uses either AzureKeyCredential (API key) or
    #    AzureCliCredential (Azure AD) to authenticate with Azure OpenAI
    #
    # 2. Model Configuration: The 'model' parameter specifies the Azure OpenAI
    #    deployment name (e.g., "gpt-4o-mini", "gpt-4", "gpt-35-turbo")
    #    This must match a deployment in your Azure OpenAI resource
    #
    # 3. Connection Management: Handles all HTTP communication with Azure
    #    OpenAI Service, including retries, rate limiting, and error handling
    #
    # 4. Agent Factory: Provides create_agent() method as a convenience for
    #    creating ChatAgent instances already bound to this client
    #
    # The client is reusable - you can create multiple agents with the same
    # client instance, and they'll all share the same connection pool and
    # authentication configuration.
    #
    model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini")
    client = AzureOpenAIChatClient(credential=credential, model=model)
    return client


def _create_retail_workflow(chat_client: AzureOpenAIChatClient):
    """Build the multi-agent retail operations workflow with store ops and supply chain agents."""

    # ========================================================================
    # CREATING AGENTS - Method 2: Using ChatAgent constructor directly
    # ========================================================================
    # This demonstrates the alternative (and more explicit) way to create agents.
    # Instead of chat_client.create_agent(), we use ChatAgent() constructor
    # and pass the chat_client as a parameter. Both methods are equivalent.
    #
    # This approach is preferred when you need more control or want to make
    # the client dependency more explicit in your code.
    #

    # Create Store Operations Analyst agent
    # This agent is specialized for in-store operations and has access to tools.
    store_ops_agent = ChatAgent(
        chat_client=chat_client,
        name="StoreOpsAnalyst",
        instructions=(
            "You own in-store operations. Combine telemetry, labor, and event data to draft an action plan "
            "covering restock, staffing, and customer experience priorities. Always call get_store_snapshot first."
        ),
        tools=[get_store_snapshot],  # This agent can call tools
    )

    # Create Supply Chain Lead agent
    # This agent specializes in supply chain coordination but has no tools.
    # It relies on information passed to it by other agents in the workflow.
    supply_chain_agent = ChatAgent(
        chat_client=chat_client,
        name="SupplyChainLead",
        instructions=(
            "You oversee supply chain for the region. Validate the store plan, commit replenishment moves, "
            "and call out any upstream dependencies for logistics or vendors."
        ),
        # Note: No tools defined - this agent works with information from handoffs
    )

    # ========================================================================
    # WORKFLOW CONSTRUCTION - Multi-Agent Orchestration
    # ========================================================================
    # WorkflowBuilder creates a directed graph of agents where each edge
    # represents a potential handoff from one agent to another.
    #
    # Key workflow methods:
    # 1. set_start_executor(agent): Defines which agent receives the initial
    #    user message and starts the workflow execution.
    #
    # 2. add_edge(from_agent, to_agent): Creates a directed edge allowing
    #    from_agent to hand off control to to_agent. The framework handles:
    #    - Context passing (conversation history)
    #    - State management
    #    - Event emission for observability
    #
    # 3. build(): Compiles the workflow graph into an executable object.
    #
    # Workflow execution flow:
    # 1. User message → StoreOpsAnalyst (start executor)
    # 2. StoreOpsAnalyst calls get_store_snapshot tool
    # 3. StoreOpsAnalyst creates initial plan
    # 4. StoreOpsAnalyst hands off to SupplyChainLead (via edge)
    # 5. SupplyChainLead receives context and adds supply chain perspective
    # 6. Workflow completes and returns all outputs
    #
    workflow = (
        WorkflowBuilder()
        .set_start_executor(store_ops_agent)      # Workflow starts here
        .add_edge(store_ops_agent, supply_chain_agent)  # Handoff edge
        .build()                                   # Compile the workflow
    )

    return workflow, store_ops_agent, supply_chain_agent


def visualize_workflow(workflow, output_dir: str = "./workflow_visualizations") -> dict[str, str]:
    """Generate workflow visualization in multiple formats.

    Returns dict with paths to generated files: {format: filepath}
    """
    if not WORKFLOW_VIZ_AVAILABLE:
        print(_warning("⚠ WorkflowViz not available. Install with: pip install agent-framework[viz]"))
        return {}

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    viz = WorkflowViz(workflow)
    generated_files = {}

    print()
    print(_heading("📊 Generating Workflow Visualizations"))

    # 1. Generate Mermaid diagram
    try:
        mermaid_content = viz.to_mermaid()
        mermaid_path = os.path.join(output_dir, "retail_ops_workflow.mmd")
        with open(mermaid_path, "w") as f:
            f.write(mermaid_content)
        generated_files["mermaid"] = mermaid_path
        print(_success(f"✓ Mermaid diagram saved: {mermaid_path}"))
        print(_muted("   View at: https://mermaid.live/"))
    except Exception as e:
        print(_warning(f"⚠ Mermaid generation failed: {e}"))

    # 2. Generate DOT (GraphViz) diagram
    try:
        dot_content = viz.to_dot()
        dot_path = os.path.join(output_dir, "retail_ops_workflow.dot")
        with open(dot_path, "w") as f:
            f.write(dot_content)
        generated_files["dot"] = dot_path
        print(_success(f"✓ GraphViz DOT saved: {dot_path}"))
    except Exception as e:
        print(_warning(f"⚠ DOT generation failed: {e}"))

    # 3. Generate SVG image
    try:
        svg_path = viz.export(os.path.join(output_dir, "retail_ops_workflow.svg"), format="svg")
        generated_files["svg"] = svg_path
        print(_success(f"✓ SVG image saved: {svg_path}"))
    except Exception as e:
        print(_warning(f"⚠ SVG export failed: {e}"))
        print(_muted("   Ensure GraphViz is installed: brew install graphviz (macOS) or apt-get install graphviz (Linux)"))

    # 4. Generate PNG image
    try:
        png_path = viz.export(os.path.join(output_dir, "retail_ops_workflow.png"), format="png")
        generated_files["png"] = png_path
        print(_success(f"✓ PNG image saved: {png_path}"))
    except Exception as e:
        print(_warning(f"⚠ PNG export failed: {e}"))

    # 5. Generate PDF
    try:
        pdf_path = viz.export(os.path.join(output_dir, "retail_ops_workflow.pdf"), format="pdf")
        generated_files["pdf"] = pdf_path
        print(_success(f"✓ PDF document saved: {pdf_path}"))
    except Exception as e:
        print(_warning(f"⚠ PDF export failed: {e}"))

    print()
    print(_heading("📁 Visualization Summary"))
    print(_kv("Output directory", output_dir))
    print(_kv("Files generated", str(len(generated_files))))
    print()

    return generated_files


async def _run_cli_demo(
    credential: AzureKeyCredential | AzureCliCredential,
    demo_type: str,
    visualize: bool = False,
    viz_dir: str = "./workflow_visualizations"
) -> None:
    """Run a specific demo scenario."""
    # Get model from environment
    model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini")

    # Create AzureOpenAIChatClient using latest syntax
    print()
    print(_heading("🤖 Initializing Azure OpenAI Chat Client"))
    chat_client = AzureOpenAIChatClient(credential=credential, model=model)
    print(_success(f"✓ Chat client initialized with model: {model}"))
    print(_muted(f"   Deployment: {model}"))
    print()

    context = EnhancedContext(chat_client=chat_client, model=model)

    # Run the selected demo scenario
    if demo_type == "tool-call":
        await run_tool_call_demo(context)
    elif demo_type == "multi-agent":
        await run_multi_agent_demo(context, visualize=visualize, viz_dir=viz_dir)
    elif demo_type == "human-loop":
        await run_human_in_loop_demo(context)
    else:
        print(_warning("⚠ No demo specified. Use --demo [tool-call|multi-agent|human-loop]"))
        print()
        print(_heading("Available Demo Scenarios:"))
        print()
        print(_colorize("1. Tool Call Demo", BOLD, FG_CYAN))
        print(_muted("   Command: python maf_enhanced.py --demo tool-call"))
        print(_muted("   Shows: Single agent using get_store_snapshot tool"))
        print()
        print(_colorize("2. Multi-Agent Demo", BOLD, FG_CYAN))
        print(_muted("   Command: python maf_enhanced.py --demo multi-agent"))
        print(_muted("   Shows: Workflow with agent handoffs (Store Ops → Supply Chain)"))
        print()
        print(_colorize("3. Human-in-the-Loop Demo", BOLD, FG_CYAN))
        print(_muted("   Command: python maf_enhanced.py --demo human-loop"))
        print(_muted("   Shows: Agent requiring human approval for critical actions"))
        print()
        print(_colorize("4. DevUI Demo", BOLD, FG_CYAN))
        print(_muted("   Command: python maf_enhanced.py --devui"))
        print(_muted("   Shows: Interactive UI for testing agents"))
        print()


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)

    load_dotenv()

    _banner("MICROSOFT AGENT FRAMEWORK – ENHANCED RETAIL OPERATIONS DEMO")
    print(_muted(f"Python {sys.version.split()[0]} on {sys.platform}"))
    print()

    _print_version_details()
    _ensure_azure_openai_env()

    # Setup observability (only for CLI mode)
    if not args.devui:
        setup_enhanced_observability()

    credential = _choose_credential()

    if args.devui:
        _launch_devui(
            credential,
            host=args.devui_host,
            port=args.devui_port,
            auto_open=not args.devui_no_auto_open,
        )
        return

    asyncio.run(_run_cli_demo(
        credential,
        demo_type=args.demo,
        visualize=args.visualize_workflow,
        viz_dir=args.viz_output_dir
    ))


if __name__ == "__main__":
    main()
