"""Semantic Kernel group chat demo using Azure OpenAI Assistants."""

import asyncio
import os
import sys
from typing import AsyncIterator, Final, Iterable

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from semantic_kernel.agents.group_chat.agent_group_chat import AgentGroupChat
from semantic_kernel.agents.open_ai.azure_assistant_agent import AzureAssistantAgent
from semantic_kernel.agents.strategies.selection.sequential_selection_strategy import SequentialSelectionStrategy
from semantic_kernel.agents.strategies.termination.default_termination_strategy import DefaultTerminationStrategy
from semantic_kernel.contents import AuthorRole


RESET = "\033[0m"
# Color palette used to distinguish roles when the terminal supports ANSI codes.
COLORS: Final[dict[str, str]] = {
    "user": "\033[95m",
    "planner": "\033[94m",
    "researcher": "\033[92m",
    "editor": "\033[93m",
    "assistant": "\033[96m",
}


DEFAULT_PROMPT: Final[str] = "Draft a lightweight product launch plan for a new AI-powered note-taking app."
ASSISTANT_NAMESPACE: Final[str] = "sk-groupchat-demo"
ASSISTANT_PROFILES: Final[tuple[tuple[str, str], ...]] = (
    (
        "planner",
        "You are the facilitator. Break the task into actionable steps, call on teammates when you hand off, "
        "and keep responses under six sentences.",
    ),
    (
        "researcher",
        "You surface real-world context or data that supports the plan. If asked for specifics you cannot access, "
        "state what assumptions you are making.",
    ),
    (
        "editor",
        "You synthesize what others proposed into a concise recommendation with next actions and open questions.",
    ),
)


def _build_client() -> tuple[AsyncAzureOpenAI, str]:
    """Create an AsyncAzureOpenAI client using environment configuration."""
    # These values point at the Azure OpenAI deployment the demo will call.
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    if not endpoint or not api_key or not deployment:
        raise ValueError(
            "Missing one of AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_CHAT_DEPLOYMENT_NAME",
        )

    client = AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    return client, deployment


def _supports_color() -> bool:
    """Return True when stdout can render ANSI colors."""
    if os.getenv("NO_COLOR") is not None:
        return False
    return sys.stdout.isatty()


def _indent(text: str) -> str:
    """Indent multiline content for readability."""
    return "\n".join(f"  {line}" if line else "  " for line in text.splitlines())


def _colorize(label: str, content: str, *, color_key: str) -> str:
    """Wrap the label and content with color and indentation when supported."""
    if not _supports_color():
        return f"{label}\n{_indent(content)}"

    color = COLORS.get(color_key, COLORS["assistant"])
    colored_label = f"{color}{label}{RESET}"
    return f"{colored_label}\n{_indent(content)}"


def _normalize_name(name: str | None) -> str:
    """Map assistant identifiers to a short display name."""
    if not name:
        return "assistant"
    if name.startswith(f"{ASSISTANT_NAMESPACE}-"):
        return name.removeprefix(f"{ASSISTANT_NAMESPACE}-")
    return name


async def _build_agents(
    *,
    client: AsyncAzureOpenAI,
    model: str,
    profiles: Iterable[tuple[str, str]],
) -> tuple[list[AzureAssistantAgent], list[str]]:
    """Create Azure Assistant agents for the supplied profiles."""
    agents: list[AzureAssistantAgent] = []
    assistant_ids: list[str] = []

    for short_name, instructions in profiles:
        # Assistant definitions are stored and executed inside Azure OpenAI.
        assistant = await client.beta.assistants.create(
            model=model,
            name=f"{ASSISTANT_NAMESPACE}-{short_name}",
            instructions=instructions,
        )
        # Semantic Kernel wraps each assistant so they can be orchestrated uniformly.
        agents.append(AzureAssistantAgent(client=client, definition=assistant))
        assistant_ids.append(assistant.id)

    return agents, assistant_ids


async def stream_group_chat(
    prompt: str,
    *,
    max_rounds: int | None = None,
) -> AsyncIterator[tuple[str, str, str]]:
    """Yield transcript entries as the multi-agent chat progresses."""
    client, deployment_name = _build_client()

    keep_created_assistants = (
        os.getenv("AZURE_OPENAI_KEEP_ASSISTANT", "false").lower() in {"true", "1", "yes"}
    )
    resolved_max_rounds = max_rounds or int(os.getenv("SEMANTIC_KERNEL_GROUPCHAT_MAX_ITERATIONS", "4"))

    agents: list[AzureAssistantAgent] = []
    assistant_ids: list[str] = []

    try:
        agents, assistant_ids = await _build_agents(
            client=client,
            model=deployment_name,
            profiles=ASSISTANT_PROFILES,
        )

        termination_strategy = DefaultTerminationStrategy(maximum_iterations=resolved_max_rounds)
        selection_strategy = SequentialSelectionStrategy()

        # AgentGroupChat is the Semantic Kernel coordinator driving multi-agent turns.
        chat = AgentGroupChat(
            agents=agents,
            termination_strategy=termination_strategy,
            selection_strategy=selection_strategy,
        )
        await chat.add_chat_message(prompt)

        yield ("user", prompt, "user")

        turn = 0
        async for message in chat.invoke():
            content = str(message).strip()
            if not content:
                continue

            role_label = message.role.value
            if message.role == AuthorRole.ASSISTANT:
                turn += 1
                display_name = _normalize_name(message.name)
                role_label = f"turn {turn}: {display_name}".strip()
                # Each assistant output is color-coded and indented for readability in the terminal.
                yield (role_label, content, display_name)
                continue

            if message.role == AuthorRole.TOOL and not content:
                continue

            # Non-assistant events (tool calls, system notices) still flow through the SK history.
            yield (role_label, content, "assistant")
    finally:
        if not keep_created_assistants:
            for assistant_id in assistant_ids:
                # Clean up Azure OpenAI resources once the demo completes.
                await client.beta.assistants.delete(assistant_id)

        await client.close()


async def generate_group_chat_transcript(
    prompt: str,
    *,
    max_rounds: int | None = None,
) -> list[tuple[str, str, str]]:
    """Return the formatted transcript for a multi-agent conversation."""
    transcript: list[tuple[str, str, str]] = []
    async for entry in stream_group_chat(prompt, max_rounds=max_rounds):
        transcript.append(entry)
    return transcript


async def run_group_chat(prompt: str) -> None:
    """Run a short round-robin group chat across multiple Azure Assistant agents."""
    transcript = await generate_group_chat_transcript(prompt)
    for label, content, color_key in transcript:
        formatted = _colorize(label, content, color_key=color_key)
        print(formatted)


def main() -> None:
    load_dotenv()
    prompt = os.getenv("SEMANTIC_KERNEL_GROUPCHAT_PROMPT", DEFAULT_PROMPT)
    asyncio.run(run_group_chat(prompt))


if __name__ == "__main__":
    main()
