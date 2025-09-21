"""Quick Semantic Kernel agent demo using Azure OpenAI Assistants."""

import asyncio
import os
from typing import Final

from dotenv import load_dotenv

from semantic_kernel.agents.group_chat.agent_group_chat import AgentGroupChat
from semantic_kernel.agents.open_ai.azure_assistant_agent import AzureAssistantAgent
from openai import AsyncAzureOpenAI
from semantic_kernel.contents import AuthorRole


DEFAULT_PROMPT: Final[str] = "Give me a concise update on the RAG concept in AI."
ASSISTANT_NAME: Final[str] = "sk-quickstart-demo"
ASSISTANT_INSTRUCTIONS: Final[str] = "You are a helpful Azure AI demo assistant. Keep answers short."  # noqa: E501


def _build_client() -> tuple[AsyncAzureOpenAI, str]:
    """Create an AsyncAzureOpenAI client using environment configuration."""
    # Values pulled from env map directly to the Azure OpenAI deployment you want to hit.
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    if not endpoint or not api_key or not deployment:
        raise ValueError("Missing one of AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

    client = AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    return client, deployment


async def run_demo(prompt: str) -> None:
    """Kick off a one-turn chat with an Azure OpenAI-powered Semantic Kernel agent."""
    client, deployment_name = _build_client()

    assistant_id = os.getenv("AZURE_OPENAI_ASSISTANT_ID")

    if assistant_id:
        assistant = await client.beta.assistants.retrieve(assistant_id)
    else:
        # The Assistant definition lives in Azure OpenAI; SK will wrap it as an agent next.
        assistant = await client.beta.assistants.create(
            model=deployment_name,
            name=ASSISTANT_NAME,
            instructions=ASSISTANT_INSTRUCTIONS,
        )

    # Semantic Kernel wraps the raw assistant definition in an agent abstraction.
    agent = AzureAssistantAgent(client=client, definition=assistant)

    # AgentGroupChat handles message history even for the single-agent case.
    chat = AgentGroupChat()
    await chat.add_chat_message(prompt)

    print(f"user: {prompt}")
    async for message in chat.invoke(agent):
        role = message.role.value
        content = str(message).strip()
        # TOOL messages echo function-call results; skip blank records for readability.
        if role == AuthorRole.TOOL.value and not content:
            continue
        print(f"{role}: {content}")

    if not assistant_id and os.getenv("AZURE_OPENAI_KEEP_ASSISTANT", "false").lower() not in {"true", "1", "yes"}:
        # Clean up the temporary Azure OpenAI assistant if we just created it.
        await client.beta.assistants.delete(assistant.id)

    await client.close()


def main() -> None:
    load_dotenv()

    prompt = os.getenv("SEMANTIC_KERNEL_DEMO_PROMPT", DEFAULT_PROMPT)

    asyncio.run(run_demo(prompt))


if __name__ == "__main__":
    main()
