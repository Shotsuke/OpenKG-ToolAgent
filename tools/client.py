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
    def __init__(self):
        """初始化MCP客户端"""
        self.exit_stack = AsyncExitStack()
        self.opanai_api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model = "qwen3-30b-a3b-instruct-2507"
        self.client = OpenAI(api_key=self.opanai_api_key, base_url=self.base_url)
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path):
        """连接到MCP服务器并列出可用工具"""
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

        console.print(
            Panel.fit(
                f"[bold green]✅ 成功连接到MCP服务器[/bold green]\n可用工具: [cyan]{[tool.name for tool in tools]}[/cyan]",
                title="[MCP Client]",
                border_style="green"
            )
        )

    async def process_query(self, query: str) -> str:
        """使用大模型处理查询并调用MCP工具"""
        messages = [{"role": "user", "content": query}]
        response = await self.session.list_tools()

        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
        } for tool in response.tools]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=available_tools,
            extra_body={"enable_thinking": False}
        )

        content = response.choices[0]

        while content.finish_reason == "tool_calls" or content.message.function_call is not None:
            tool_call = content.message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            console.print(
                Panel.fit(
                    f"[bold yellow]工具调用:[/bold yellow] [cyan]{tool_name}[/cyan]\n"
                    f"[bold]参数:[/bold] {json.dumps(tool_args, ensure_ascii=False, indent=2)}",
                    title="[LLM → MCP Tool]",
                    border_style="yellow"
                )
            )

            result = await self.session.call_tool(tool_name, tool_args)

            console.print(
                Panel.fit(
                    f"{result.content[0].text}",
                    title=f"[返回结果] {tool_name}",
                    border_style="blue"
                )
            )

            messages.append(content.message.model_dump())
            messages.append({
                "role": "tool",
                "content": result.content[0].text,
                "tool_call_id": tool_call.id,
            })

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body={"enable_thinking": False}
            )
            content = response.choices[0]

        return content.message.content

    async def chat_loop(self):
        """运行交互式聊天"""
        console.print(
            Panel.fit(
                "MCP客户端已启动！输入 [bold red]quit[/bold red] 退出",
                title="[欢迎]",
                border_style="magenta"
            )
        )

        while True:
            try:
                query = Prompt.ask("[bold cyan]用户[/bold cyan]")
                if query.lower() == 'quit':
                    break
                response = await self.process_query(query)
                console.print(
                    Panel.fit(
                        f"{response}",
                        title="[LLM Agent]",
                        border_style="cyan"
                    )
                )
            except Exception as e:
                console.print(
                    Panel.fit(
                        f"发生错误: {str(e)}",
                        title="[Error]",
                        border_style="red"
                    )
                )

    async def clean(self):
        """清理资源"""
        await self.exit_stack.aclose()
