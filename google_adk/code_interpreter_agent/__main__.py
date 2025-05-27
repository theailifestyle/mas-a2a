import asyncio
import functools
import logging
import os
import sys

import click
import uvicorn

from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from adk_agent_executor import ADKCodeInterpreterAgentExecutor


load_dotenv()

logging.basicConfig()


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10010)
def main(host: str, port: int):
    if not os.getenv('GOOGLE_API_KEY'):
        print('GOOGLE_API_KEY environment variable not set.')
        sys.exit(1)

    skill = AgentSkill(
        id='code_interpreter',
        name='Code Interpreter',
        description='Executes Python code to perform calculations and data manipulation.',
        tags=['code', 'calculation', 'data manipulation'],
        examples=[
            'Calculate the value of (5 + 7) * 3',
            'What is 10 factorial?',
            'Solve for x: 2x + 5 = 15',
        ],
    )

    agent_executor = ADKCodeInterpreterAgentExecutor()
    agent_card = AgentCard(
        name='ADK Code Interpreter Agent',
        description='I can execute Python code to perform calculations and data manipulation.',
        url=f'http://{host}:{port}/',
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, task_store=InMemoryTaskStore()
    )
    app = A2AStarletteApplication(agent_card, request_handler)
    uvicorn.run(app.build(), host=host, port=port)


if __name__ == '__main__':
    main()
