# import asyncio
# import json
# import os
# from typing import Optional
# from contextlib import AsyncExitStack
# from openai import OpenAI
# from dotenv import load_dotenv

# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client

# from rich.console import Console
# from rich.panel import Panel
# from rich.prompt import Prompt

# console = Console()
# load_dotenv()

# class MCPClient:
#     def __init__(self):
#         """初始化MCP客户端"""
#         self.exit_stack = AsyncExitStack()
#         self.opanai_api_key = os.getenv("DASHSCOPE_API_KEY")
#         self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
#         self.model = "qwen3-30b-a3b-instruct-2507"
#         self.client = OpenAI(api_key=self.opanai_api_key, base_url=self.base_url)
#         self.session: Optional[ClientSession] = None
#         self.exit_stack = AsyncExitStack()

#     async def connect_to_server(self, server_script_path):
#         """连接到MCP服务器并列出可用工具"""
#         server_params = StdioServerParameters(
#             command="python",
#             args=[server_script_path],
#             env=None
#         )
#         stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
#         self.stdio, self.write = stdio_transport
#         self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

#         await self.session.initialize()
#         response = await self.session.list_tools()
#         tools = response.tools

#         console.print(
#             Panel.fit(
#                 f"[bold green]✅ 成功连接到MCP服务器[/bold green]\n可用工具: [cyan]{[tool.name for tool in tools]}[/cyan]",
#                 title="[MCP Client]",
#                 border_style="green"
#             )
#         )

#     async def process_query(self, query: str) -> str:
#         """使用大模型处理查询并调用MCP工具"""
#         messages = [{"role": "user", "content": query}]
#         response = await self.session.list_tools()

#         available_tools = [{
#             "type": "function",
#             "function": {
#                 "name": tool.name,
#                 "description": tool.description,
#                 "input_schema": tool.inputSchema
#             }
#         } for tool in response.tools]

#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=messages,
#             tools=available_tools,
#             extra_body={"enable_thinking": False}
#         )

#         content = response.choices[0]

#         while content.finish_reason == "tool_calls" or content.message.function_call is not None:
#             tool_call = content.message.tool_calls[0]
#             tool_name = tool_call.function.name
#             tool_args = json.loads(tool_call.function.arguments)

#             console.print(
#                 Panel.fit(
#                     f"[bold yellow]工具调用:[/bold yellow] [cyan]{tool_name}[/cyan]\n"
#                     f"[bold]参数:[/bold] {json.dumps(tool_args, ensure_ascii=False, indent=2)}",
#                     title="[LLM → MCP Tool]",
#                     border_style="yellow"
#                 )
#             )

#             result = await self.session.call_tool(tool_name, tool_args)

#             console.print(
#                 Panel.fit(
#                     f"{result.content[0].text}",
#                     title=f"[返回结果] {tool_name}",
#                     border_style="blue"
#                 )
#             )

#             messages.append(content.message.model_dump())
#             messages.append({
#                 "role": "tool",
#                 "content": result.content[0].text,
#                 "tool_call_id": tool_call.id,
#             })

#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=messages,
#                 extra_body={"enable_thinking": False}
#             )
#             content = response.choices[0]

#         return content.message.content

#     async def chat_loop(self):
#         """运行交互式聊天"""
#         console.print(
#             Panel.fit(
#                 "MCP客户端已启动！输入 [bold red]quit[/bold red] 退出",
#                 title="[欢迎]",
#                 border_style="magenta"
#             )
#         )

#         while True:
#             try:
#                 query = Prompt.ask("[bold cyan]用户[/bold cyan]")
#                 if query.lower() == 'quit':
#                     break
#                 response = await self.process_query(query)
#                 console.print(
#                     Panel.fit(
#                         f"{response}",
#                         title="[LLM Agent]",
#                         border_style="cyan"
#                     )
#                 )
#             except Exception as e:
#                 console.print(
#                     Panel.fit(
#                         f"发生错误: {str(e)}",
#                         title="[Error]",
#                         border_style="red"
#                     )
#                 )

#     async def clean(self):
#         """清理资源"""
#         await self.exit_stack.aclose()

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
        self.model = "qwen-plus-2025-04-28"
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

        console.print("[bold green]>>> ✅ 成功连接到MCP服务器 <<<[/bold green]")
        if tools:
            console.print("[bold cyan]可用工具列表:[/bold cyan]")
            for tool in tools:
                console.print(f"  [bold yellow]• {tool.name}[/bold yellow]")
        else:
            console.print("[dim]（无可用工具）[/dim]")

    async def process_query(self, query: str) -> str:
        """使用大模型处理查询并调用MCP工具"""
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
            {
                "role": "user", 
                "content": query
            }
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

        stream_response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=available_tools,
            extra_body={"enable_thinking": True},
            parallel_tool_calls=True,
            stream = True
        )
        
        answer_content = ""

        # 回复为空，要么还没进行回答，要么还在请求调用工具。得到回复后即可返回
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
                stream = True
            )

            for chunk in stream_response:
                if not chunk.choices:
                    console.print(f"\n[dim]Usage: {chunk.usage}[/dim]")
                    
                delta = chunk.choices[0].delta
                # print(delta)
                # Reasoning
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    reasoning_content += delta.reasoning_content
                    print(delta.reasoning_content,end="",flush=True) 

                else:
                    if not is_answering:  # 首次进入回复阶段时打印标题
                        is_answering = True
                        console.print("\n[bold cyan]==== 回复内容 ====[/bold cyan]")
                    if delta.content is not None:
                        answer_content += delta.content
                        print(delta.content,end="",flush=True)  # 流式输出回复内容

                    if delta.tool_calls is not None:
                        for tool_call in delta.tool_calls:
                            index = tool_call.index  # 工具调用索引，用于并行调用
                            
                            # 动态扩展工具信息存储列表
                            while len(tool_info) <= index:
                                tool_info.append({})
                            
                            # 收集工具调用ID（用于后续函数调用）
                            if tool_call.id:
                                tool_info[index]['id'] = tool_info[index].get('id', '') + tool_call.id
                            
                            # 收集函数名称（用于后续路由到具体函数）
                            if tool_call.function and tool_call.function.name:
                                tool_info[index]['name'] = tool_info[index].get('name', '') + tool_call.function.name
                            
                            # 收集函数参数（JSON字符串格式，需要后续解析）
                            if tool_call.function and tool_call.function.arguments:
                                tool_info[index]['arguments'] = tool_info[index].get('arguments', '') + tool_call.function.arguments

            # print(f"\n"+"="*19+"工具调用信息"+"="*19)
            if tool_info:
                console.print("\n[bold cyan]==== 工具调用 ====[/bold cyan]")
                for tool in tool_info:
                    # tool_id = tool['id']
                    tool_name = tool['name']
                    tool_args = json.loads(tool['arguments']) if tool['arguments'] else {}

                    result = await self.session.call_tool(tool_name, tool_args)
                    tool_result = result.content[0].text if result.content else "无结果"
                    tool['result'] = tool_result

                    console.print(f"[bold yellow]工具名称:[/bold yellow] {tool_name}")
                    console.print(f"[bold magenta]调用参数:[/bold magenta] {json.dumps(tool_args, ensure_ascii=False, indent=2)}")
                    console.print(f"[bold green]返回结果:[/bold green] {tool_result}\n")
                    
                    # 添加工具调用到消息
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

                    # 添加工具结果到消息
                    messages.append({
                        "role": "tool",
                        "content": json.dumps({"result": tool_result}, ensure_ascii=False),
                        "tool_call_id": tool['id']
                    })

                # messages.extend(tool_calls_messages)
                # messages.extend(tool_results_messages)

        return answer_content

    async def chat_loop(self):
        """运行交互式聊天"""
        console.print("[bold magenta]>>> MCP客户端已启动！输入 [bold red]quit[/bold red] 退出 <<<[/bold magenta]")

        while True:
            try:
                query = Prompt.ask("[bold cyan]用户[/bold cyan]")
                if query.lower() == 'quit':
                    break
                response = await self.process_query(query)
                console.print(
                    f"\n\n[bold green][LLM Agent][/bold green] {response}\n"
                )
            except Exception as e:
                console.print(f"[bold black]>>> ERROR: 发生错误: {str(e)} <<<[/bold black]")

    async def clean(self):
        """清理资源"""
        await self.exit_stack.aclose()



