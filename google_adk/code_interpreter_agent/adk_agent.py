# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.adk.agents.llm_agent import Agent
from google.adk.code_executors import BuiltInCodeExecutor
from google.genai import types


async def create_code_interpreter_agent() -> Agent:
    """Constructs the ADK code interpreter agent."""
    return Agent(
        model='gemini-2.0-flash',
        name='code_interpreter_agent_adk',
        description='An agent that can execute Python code to perform calculations.',
        instruction="""You are a calculator agent.
        When given a mathematical expression, write and execute Python code to calculate the result.
        Return only the final numerical result as plain text, without markdown or code blocks.
        """,
        code_executor=BuiltInCodeExecutor(),
    )
