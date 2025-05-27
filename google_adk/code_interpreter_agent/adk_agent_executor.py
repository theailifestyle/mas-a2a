# mypy: ignore-errors
import asyncio
import logging

from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any
from uuid import uuid4

from google.adk import Runner
from google.adk.agents.llm_agent import Agent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.events import Event
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.adk.tools import BaseTool, ToolContext
from google.genai import types
from pydantic import ConfigDict

from a2a.client import A2AClient
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Artifact,
    FilePart,
    FileWithBytes,
    FileWithUri,
    GetTaskRequest,
    GetTaskSuccessResponse,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    SendMessageSuccessResponse,
    Task,
    TaskQueryParams,
    TaskState,
    TaskStatus,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import get_text_parts
from a2a.utils.errors import ServerError

from adk_agent import create_code_interpreter_agent


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ADKCodeInterpreterAgentExecutor(AgentExecutor):
    """An AgentExecutor that runs an ADK-based Code Interpreter Agent."""

    def __init__(self):
        self._agent = asyncio.run(create_code_interpreter_agent())
        self.runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    def _run_agent(
        self,
        session_id,
        new_message: types.Content,
        task_updater: TaskUpdater,
    ) -> AsyncGenerator[Event]:
        return self.runner.run_async(
            session_id=session_id,
            user_id='self',
            new_message=new_message,
        )

    async def _process_request(
        self,
        new_message: types.Content,
        session_id: str,
        task_updater: TaskUpdater,
    ) -> AsyncIterable[TaskStatus | Artifact]:
        session = await self._upsert_session(
            session_id,
        )
        session_id = session.id
        async for event in self._run_agent(
            session_id, new_message, task_updater
        ):
            logger.debug('Received ADK event: %s', event)
            if event.content and event.content.parts:
                # Process all parts for logging/debugging
                for part in event.content.parts:
                    if part.executable_code:
                        logger.debug(
                            'Agent generated code:\n```python\n%s\n```',
                            part.executable_code.code,
                        )
                    elif part.code_execution_result:
                        logger.debug(
                            'Code Execution Result: %s - Output:\n%s',
                            part.code_execution_result.outcome,
                            part.code_execution_result.output,
                        )
                    elif part.text and not part.text.isspace():
                        logger.debug("Text: '%s'", part.text.strip())

                # Only process the event if it's the final one
                if event.is_final_response():
                    if event.content and event.content.parts:
                        response_parts = convert_genai_parts_to_a2a(event.content.parts)
                        logger.debug('Yielding final response: %s', response_parts)
                        
                        logger.debug('Yielding final response parts: %s', response_parts)
                        
                        # Add artifact if there are parts
                        if response_parts:
                            task_updater.add_artifact(response_parts)

                        # Call complete() without a message, like search_agent
                        task_updater.complete() 
                    else:
                        # Final response but no content/parts, complete with an empty message or error
                        logger.warning("Final response received with no content/parts.")
                        task_updater.complete() 
                    break # Exit loop after final response
                # Skip all other intermediate events
                else:
                    logger.debug('Skipping intermediate event: %s', event)
            else:
                logger.debug('Skipping event (no content or parts)')

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ):
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            updater.submit()
        updater.start_work()
        await self._process_request(
            types.UserContent(
                parts=convert_a2a_parts_to_genai(context.message.parts),
            ),
            context.context_id,
            updater,
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise ServerError(error=UnsupportedOperationError())

    async def _upsert_session(self, session_id: str):
        return await self.runner.session_service.get_session(
            app_name=self.runner.app_name, user_id='self', session_id=session_id
        ) or await self.runner.session_service.create_session(
            app_name=self.runner.app_name, user_id='self', session_id=session_id
        )


def convert_a2a_parts_to_genai(parts: list[Part]) -> list[types.Part]:
    """Convert a list of A2A Part types into a list of Google Gen AI Part types."""
    return [convert_a2a_part_to_genai(part) for part in parts]


def convert_a2a_part_to_genai(part: Part) -> types.Part:
    """Convert a single A2A Part type into a Google Gen AI Part type."""
    part = part.root
    if isinstance(part, TextPart):
        return types.Part(text=part.text)
    if isinstance(part, FilePart):
        if isinstance(part.file, FileWithUri):
            return types.Part(
                file_data=types.FileData(
                    file_uri=part.file.uri, mime_type=part.file.mime_type
                )
            )
        if isinstance(part.file, FileWithBytes):
            return types.Part(
                inline_data=types.Blob(
                    data=part.file.bytes, mime_type=part.file.mime_type
                )
            )
        raise ValueError(f'Unsupported file type: {type(part.file)}')
    raise ValueError(f'Unsupported part type: {type(part)}')


def convert_genai_parts_to_a2a(parts: list[types.Part]) -> list[Part]:
    """Convert a list of Google Gen AI Part types into a list of A2A Part types."""
    # Revert to the version that only returns the single final text part for stability
    for part_item in parts:
        if part_item.text and not part_item.text.isspace():
            # Ensure we are returning a list containing a Part object
            return [TextPart(text=part_item.text.strip())] 
    return [] # If no text part found, return empty list


def convert_genai_part_to_a2a(part: types.Part) -> Part:
    """Convert a single Google Gen AI Part type into an A2A Part type.
    This function is effectively bypassed by the simplified convert_genai_parts_to_a2a for now."""
    if part.text:
        return TextPart(text=part.text.strip())
    # Keeping the original logic here for completeness if we revert fully later.
    if part.executable_code:
        return TextPart(text=f"Agent generated code:\n```python\n{part.executable_code.code}\n```")
    if part.code_execution_result:
        return TextPart(text=f"Code Execution Result: {part.code_execution_result.outcome} - Output:\n{part.code_execution_result.output}")
    if part.file_data:
        return FilePart(
            file=FileWithUri(
                uri=part.file_data.file_uri,
                mime_type=part.file_data.mime_type,
            )
        )
    if part.inline_data:
        return Part(
            root=FilePart(
                file=FileWithBytes(
                    bytes=part.inline_data.data,
                    mime_type=part.inline_data.mime_type,
                )
            )
        )
    raise ValueError(f'Unsupported part type: {part}')
