"""MCPClient 模块：异步大模型客户端与工具调用控制器。

该模块实现了一个异步 MCP (Model Context Protocol) 客户端，
能够连接到自定义 MCP 工具服务器，并使用大模型（如 Qwen 系列）
进行自然语言处理、自动工具调用与流式推理输出。

主要功能：
    - 自动连接 MCP 工具服务器并列出可用工具；
    - 通过 LLM (Qwen) 流式推理生成响应；
    - 支持 `parallel_tool_calls` 并行工具调用；
    - 自动记录 reasoning（思考过程）与工具调用链；
    - 提供交互式命令行聊天界面。

环境变量要求：
    - DASHSCOPE_API_KEY: 阿里 DashScope 平台 API Key。
"""

import asyncio
import json
import os
from typing import Optional
from contextlib import AsyncExitStack
from openai import OpenAI
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()
load_dotenv()


class MCPClient:
    """异步 MCP 客户端类。

    提供连接 MCP 工具服务器、执行大模型推理、调用工具函数、
    并以流式形式输出 reasoning 与最终回答。

    Attributes:
        exit_stack (AsyncExitStack): 异步上下文堆栈，用于资源清理。
        opanai_api_key (str): DashScope 平台 API Key。
        base_url (str): DashScope 兼容 OpenAI SDK 的 API 地址。
        model (str): 使用的模型名称（默认为 Qwen Plus 2025）。
        client (OpenAI): OpenAI 客户端对象，用于发送请求。
        session (Optional[ClientSession]): 当前 MCP 客户端会话对象。
    """

    def __init__(self):
        """初始化 MCP 客户端。"""
        self.exit_stack = AsyncExitStack()
        self.opanai_api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model = "qwen-plus-2025-04-28"
        self.client = OpenAI(api_key=self.opanai_api_key, base_url=self.base_url)
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path: str):
        """连接到 MCP 服务器并列出可用工具。

        Args:
            server_script_path (str): MCP 工具服务器脚本路径。

        Raises:
            Exception: 当无法连接或初始化 MCP 会话时触发。
        """
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()
        response = await self.session.list_tools()
        tools = response.tools

        console.print("[bold green]>>> ✅ 成功连接到MCP服务器 <<<[/bold green]")
        if tools:
            console.print("[bold cyan]可用工具列表:[/bold cyan]")
            for tool in tools:
                console.print(f"  [bold yellow]• {tool.name}[/bold yellow]")
        else:
            console.print("[dim]（无可用工具）[/dim]")

    async def process_query(self, query: str) -> str:
        """使用大模型处理查询并自动调用 MCP 工具。

        Args:
            query (str): 用户输入的自然语言查询。

        Returns:
            str: 模型的最终自然语言回复。

        Notes:
            - 本函数在推理阶段会实时输出「思考过程」与「回复内容」。
            - 当检测到模型生成的 tool_calls 时，会自动调用 MCP 工具。
            - 工具返回结果将写回消息上下文，用于下一轮模型推理。
        """
        messages = [
            {
                "role": "system",
                "content": """你是一个很有帮助的助手。
                对于能够调用工具函数并行解决的问题，你可以一次性给出需要的函数调用。
                如果你认为某个问题可以被你很轻松地解决，不需要经过工具调用，你也可以选择直接跳过。
                如果你遇到了明显需要分步调用工具函数的问题，请你逐步来，不必一口气解决。
                在这种情况下，你会被多次调用，你的上一次对工具的调用会被记录，然后输入给你的下一次调用，
                直到不需要调用工具，最终再来输出回答。
                这份工具调用将严格按照时间顺序来记录，方便你分析前后顺序。
                请以友好的语气回答问题。""",
            },
            {"role": "user", "content": query}
        ]

        response = await self.session.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
        } for tool in response.tools]

        answer_content = ""

        while not answer_content:
            console.print("\n[bold cyan]==== 思考过程 ====[/bold cyan]")
            reasoning_content = ""
            is_answering = False
            tool_info = []

            stream_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=available_tools,
                extra_body={"enable_thinking": True},
                parallel_tool_calls=True,
                stream=True
            )

            for chunk in stream_response:
                if not chunk.choices:
                    console.print(f"\n[dim]Usage: {chunk.usage}[/dim]")
                    continue

                delta = chunk.choices[0].delta

                # 输出模型思考过程
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    reasoning_content += delta.reasoning_content
                    print(delta.reasoning_content, end="", flush=True)

                # 输出最终回答与工具调用信息
                else:
                    if not is_answering:
                        is_answering = True
                        console.print("\n[bold cyan]==== 回复内容 ====[/bold cyan]")

                    if delta.content is not None:
                        answer_content += delta.content
                        print(delta.content, end="", flush=True)

                    if delta.tool_calls is not None:
                        for tool_call in delta.tool_calls:
                            index = tool_call.index
                            while len(tool_info) <= index:
                                tool_info.append({})

                            if tool_call.id:
                                tool_info[index]['id'] = tool_info[index].get('id', '') + tool_call.id
                            if tool_call.function and tool_call.function.name:
                                tool_info[index]['name'] = tool_info[index].get('name', '') + tool_call.function.name
                            if tool_call.function and tool_call.function.arguments:
                                tool_info[index]['arguments'] = tool_info[index].get('arguments', '') + tool_call.function.arguments

            # 执行工具调用
            if tool_info:
                console.print("\n[bold cyan]==== 工具调用 ====[/bold cyan]")
                for tool in tool_info:
                    tool_name = tool['name']
                    tool_args = json.loads(tool['arguments']) if tool['arguments'] else {}

                    result = await self.session.call_tool(tool_name, tool_args)
                    tool_result = result.content[0].text if result.content else "无结果"
                    tool['result'] = tool_result

                    console.print(f"[bold yellow]工具名称:[/bold yellow] {tool_name}")
                    console.print(f"[bold magenta]调用参数:[/bold magenta] {json.dumps(tool_args, ensure_ascii=False, indent=2)}")
                    console.print(f"[bold green]返回结果:[/bold green] {tool_result}\n")

                    # 将调用及结果添加回对话上下文
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool['id'],
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args, ensure_ascii=False)
                            }
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "content": json.dumps({"result": tool_result}, ensure_ascii=False),
                        "tool_call_id": tool['id']
                    })

        return answer_content

    async def chat_loop(self):
        """启动交互式聊天循环。

        用户可在命令行中输入查询，模型将输出流式 reasoning 与回答。
        输入 'quit' 即可退出程序。
        """
        console.print("[bold magenta]>>> MCP客户端已启动！输入 [bold red]quit[/bold red] 退出 <<<[/bold magenta]")

        while True:
            try:
                query = Prompt.ask("[bold cyan]用户[/bold cyan]")
                if query.lower() == 'quit':
                    break
                response = await self.process_query(query)
                console.print(f"\n\n[bold green][LLM Agent][/bold green] {response}\n")
            except Exception as e:
                console.print(f"[bold black]>>> ERROR: 发生错误: {str(e)} <<<[/bold black]")

    async def clean(self):
        """关闭所有异步上下文并清理资源。"""
        await self.exit_stack.aclose()
