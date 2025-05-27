import asyncio
import argparse
import httpx
from uuid import uuid4
from a2a.client import A2AClient
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
    SendMessageSuccessResponse,
    JSONRPCErrorResponse,
    Task,
    Message,
)

async def main():
    parser = argparse.ArgumentParser(description="CLI client for the Code Interpreter Agent.")
    parser.add_argument("--query", type=str, required=True, help="The mathematical expression or code query.")
    args = parser.parse_args()

    agent_url = "http://localhost:10010"

    async with httpx.AsyncClient() as httpx_client:
        client = A2AClient(url=agent_url, httpx_client=httpx_client)

        print(f"Sending query '{args.query}' to agent at {agent_url}...")

        try:
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': args.query}
                    ],
                    'messageId': uuid4().hex,
                },
                'agentId': "code_interpreter_agent",
                'userId': "cli_user",
            }
            request = SendMessageRequest(
                params=MessageSendParams(**send_message_payload)
            )

            response = await client.send_message(request)
            print("\nRaw Agent Response:")
            print(response.model_dump_json(indent=2)) # Print full raw response

            print("\nParsed Agent Response:")
            if isinstance(response.root, SendMessageSuccessResponse):
                if isinstance(response.root.result, Task):
                    if response.root.result.artifacts:
                        for artifact in response.root.result.artifacts:
                            if artifact.parts:
                                for part in artifact.parts:
                                    # Print all parts, not just text
                                    print(f"  Part Kind: {part.root.kind}")
                                    if part.root.kind == 'text':
                                        print(f"    Text: {part.root.text}")
                                    elif part.root.kind == 'file':
                                        if hasattr(part.root.file, 'uri'):
                                            print(f"    File URI: {part.root.file.uri}")
                                        elif hasattr(part.root.file, 'bytes'):
                                            print(f"    File Bytes (truncated): {part.root.file.bytes[:50]}...")
                                    elif part.root.kind == 'data':
                                        print(f"    Data: {part.root.data}")
                    else:
                        print("No artifacts found in agent response.")
                elif isinstance(response.root.result, Message):
                    if response.root.result.parts:
                        for part in response.root.result.parts:
                            # Print all parts, not just text
                            print(f"  Part Kind: {part.root.kind}")
                            if part.root.kind == 'text':
                                print(f"    Text: {part.root.text}")
                            elif part.root.kind == 'file':
                                if hasattr(part.root.file, 'uri'):
                                    print(f"    File URI: {part.root.file.uri}")
                                elif hasattr(part.root.file, 'bytes'):
                                    print(f"    File Bytes (truncated): {part.root.file.bytes[:50]}...")
                            elif part.root.kind == 'data':
                                print(f"    Data: {part.root.data}")
                    else:
                        print("No parts found in agent response message.")
                else:
                    print(f"Unexpected result type: {type(response.root.result)}")
            elif isinstance(response.root, JSONRPCErrorResponse):
                print(f"Agent returned an error: {response.root.error.message}")
            else:
                print("Could not parse agent response: Unknown response type.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
