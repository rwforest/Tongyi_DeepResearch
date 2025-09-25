"""
Modified React Agent for Databricks
Replaces OpenAI SDK calls with predict() function
Based on inference/react_agent.py
"""

import json
import json5
import os
import time
import random
import datetime
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union, Callable

from qwen_agent.llm.schema import Message
from qwen_agent.utils.utils import build_text_completion_prompt
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from qwen_agent.utils.utils import format_as_text_message, merge_generate_cfgs

# Import tools (now in same directory)
from tool_file import *
from tool_scholar import *
from tool_python import *
from tool_search import *
from tool_visit import *
from prompt import *

OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'

MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 100))

TOOL_CLASS = [
    FileParser(),
    Scholar(),
    Visit(),
    Search(),
    PythonInterpreter(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")


class DatabricksMultiTurnReactAgent(FnCallAgent):
    """
    Modified MultiTurnReactAgent that uses predict() function instead of OpenAI SDK
    """

    def __init__(self,
                 predict_function: Callable[[str], str],
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 **kwargs):
        """
        Initialize with predict function instead of LLM config

        Args:
            predict_function: Your predict() function that takes a string and returns response
            function_list: List of available functions/tools
            llm: LLM configuration (for compatibility, but we'll use predict_function)
        """
        self.predict_function = predict_function
        self.llm_generate_cfg = llm.get("generate_cfg", {}) if llm else {}
        self.llm_local_path = llm.get("model", "databricks-model") if llm else "databricks-model"
        self.model = self.llm_local_path

        print(f"✅ Initialized DatabricksMultiTurnReactAgent with predict() function")
        print(f"   Available tools: {list(TOOL_MAP.keys())}")

    def sanity_check_output(self, content):
        return "<think>" in content and "</think>" in content

    def call_predict(self, msgs, max_tries=3):
        """
        Use predict() function instead of OpenAI API

        Args:
            msgs: List of message dictionaries
            max_tries: Maximum retry attempts

        Returns:
            Generated response string
        """
        # Convert messages to prompt format for predict function
        prompt_parts = []

        for msg in msgs:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        # Add final assistant prompt
        prompt_parts.append("Assistant:")
        full_prompt = "\n\n".join(prompt_parts)

        # Retry logic
        for attempt in range(max_tries):
            try:
                print(f"--- Calling predict() function, attempt {attempt + 1}/{max_tries} ---")

                # Call your predict function
                response = self.predict_function(full_prompt)

                if response and response.strip():
                    print("--- Predict function call successful ---")
                    return response.strip()
                else:
                    print(f"Warning: Attempt {attempt + 1} received empty response")

            except Exception as e:
                print(f"Error: Attempt {attempt + 1} failed: {e}")

            if attempt < max_tries - 1:
                sleep_time = 1 * (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)

        raise Exception(f"Predict function failed after {max_tries} attempts")

    def _run(self, data, model=None, planning_port=None):
        """
        Main run method - modified to use predict() function instead of server calls

        Args:
            data: Task data containing question and answer
            model: Model name (for compatibility)
            planning_port: Port (not used, for compatibility)

        Returns:
            Result dictionary with prediction and metadata
        """
        start_time = time.time()
        question = data['item']['question']
        answer = data['item']['answer']

        print(f"🔬 Processing question: {question[:100]}...")

        self.user_prompt = question
        system_prompt = SYSTEM_PROMPT
        cur_date = today_date()
        system_prompt = system_prompt + str(cur_date)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round_count = 0

        while num_llm_calls_available > 0:
            # Check time limit (150 minutes)
            if time.time() - start_time > 150 * 60:
                prediction = 'No answer found after 2h30mins'
                termination = 'No answer found after 2h30mins'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                return result

            round_count += 1
            num_llm_calls_available -= 1

            # Call predict function instead of server
            content = self.call_predict(messages)

            print(f'Round {round_count}: {content}')

            # Remove tool_response if present in content
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]

            messages.append({"role": "assistant", "content": content.strip()})

            # Check for tool calls
            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]

                try:
                    # Handle Python code specially
                    if "python" in tool_call.lower():
                        try:
                            code_raw = content.split('<tool_call>')[1].split('</tool_call>')[0].split('<code>')[1].split('</code>')[0].strip()
                            result = TOOL_MAP['PythonInterpreter'].call(code_raw)
                        except:
                            result = "[Python Interpreter Error]: Formatting error."
                    else:
                        # Parse JSON tool call
                        tool_call_parsed = json5.loads(tool_call)
                        tool_name = tool_call_parsed.get('name', '')
                        tool_args = tool_call_parsed.get('arguments', {})
                        result = self.custom_call_tool(tool_name, tool_args)

                except Exception as e:
                    result = f'Error: Tool call is not valid JSON or tool execution failed: {str(e)}'

                # Add tool response
                tool_response = f"<tool_response>\n{result}\n</tool_response>"
                messages.append({"role": "user", "content": tool_response})

                print(f"Tool executed: {result[:200]}..." if len(str(result)) > 200 else f"Tool result: {result}")

            # Check for final answer
            if '<answer>' in content and '</answer>' in content:
                termination = 'answer'
                break

            # Check if we're out of calls
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            # Check token count (simplified)
            max_tokens = 108 * 1024
            token_count = self.count_tokens(messages)
            print(f"round: {round_count}, token count: {token_count}")

            if token_count > max_tokens:
                print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")

                messages[-1]['content'] = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:<think>your final thinking</think>\n<answer>your answer</answer>"

                content = self.call_predict(messages)
                messages.append({"role": "assistant", "content": content.strip()})

                if '<answer>' in content and '</answer>' in content:
                    prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                    termination = 'generate an answer as token limit reached'
                else:
                    prediction = messages[-1]['content']
                    termination = 'format error: generate an answer as token limit reached'

                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                return result

        # Extract final prediction
        if '<answer>' in messages[-1]['content']:
            prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'

        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination
        }
        return result

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        """Execute a tool with given arguments"""
        if tool_name in TOOL_MAP:
            try:
                return TOOL_MAP[tool_name].call(tool_args)
            except Exception as e:
                return f"Tool {tool_name} execution failed: {str(e)}"
        else:
            return f"Tool {tool_name} not found in available tools: {list(TOOL_MAP.keys())}"

    def count_tokens(self, messages):
        """
        Simplified token counting - just count characters and divide by 4
        For production, you might want to use a proper tokenizer
        """
        total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
        return total_chars // 4  # Rough approximation