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
from typing import Optional, List, TypedDict, Annotated
from contextlib import AsyncExitStack
from openai import AsyncOpenAI
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

console = Console()
load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


class MCPClient:
    """异步 MCP 客户端类。

    提供连接 MCP 工具服务器、执行大模型推理、调用工具函数、
    并以流式形式输出 reasoning 与最终回答。

    Attributes:
        exit_stack (AsyncExitStack): 异步上下文堆栈，用于资源清理。
        opanai_api_key (str): DashScope 平台 API Key。
        base_url (str): DashScope 兼容 OpenAI SDK 的 API 地址。
        model (str): 使用的模型名称（默认为 Qwen Plus 2025）。
        client (AsyncOpenAI): OpenAI 异步客户端对象，用于发送请求。
        session (Optional[ClientSession]): 当前 MCP 客户端会话对象。
    """

    def __init__(self):
        """初始化 MCP 客户端。"""
        self.exit_stack = AsyncExitStack()
        self.opanai_api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model = os.getenv("MODEL")
        self.client = AsyncOpenAI(api_key=self.opanai_api_key, base_url=self.base_url)
        self.session: Optional[ClientSession] = None
        self.tools_schema = []
        self.app = None

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

        # 转换工具定义供 OpenAI 使用
        self.tools_schema = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
        } for tool in tools]

        # 构建 LangGraph
        self.build_graph()

        console.print("[bold green]>>> ✅ 成功连接到MCP服务器 <<<[/bold green]")
        if tools:
            console.print("[bold cyan]可用工具列表:[/bold cyan]")
            for tool in tools:
                console.print(f"  [bold yellow]• {tool.name}[/bold yellow]")
        else:
            console.print("[dim]（无可用工具）[/dim]")

    def build_graph(self):
        """构建 LangGraph 工作流"""
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.call_tools)

        workflow.set_entry_point("agent")

        def should_continue(state: AgentState):
            messages = state['messages']
            last_message = messages[-1]
            if last_message.tool_calls:
                return "tools"
            return END

        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                END: END
            }
        )
        workflow.add_edge("tools", "agent")

        self.app = workflow.compile()

    async def call_model(self, state: AgentState):
        """调用大模型节点"""
        messages = state['messages']
        
        # 将 LangChain 消息转换为 OpenAI 格式
        openai_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                m = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    m["tool_calls"] = [{
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"])
                        }
                    } for tc in msg.tool_calls]
                openai_messages.append(m)
            elif isinstance(msg, ToolMessage):
                openai_messages.append({
                    "role": "tool", 
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id
                })

        console.print("\n[bold cyan]==== 思考过程 ====[/bold cyan]")
        reasoning_content = ""
        answer_content = ""
        is_answering = False
        tool_calls_accumulated = []
        
        stream_response = await self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            tools=self.tools_schema if self.tools_schema else None,
            extra_body={"enable_thinking": True},
            parallel_tool_calls=True,
            stream=True
        )

        async for chunk in stream_response:
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
                if not is_answering and (delta.content or delta.tool_calls):
                    is_answering = True
                    console.print("\n[bold cyan]==== 回复内容 ====[/bold cyan]")

                if delta.content is not None:
                    answer_content += delta.content
                    print(delta.content, end="", flush=True)

                if delta.tool_calls is not None:
                    for tool_call in delta.tool_calls:
                        index = tool_call.index
                        while len(tool_calls_accumulated) <= index:
                            tool_calls_accumulated.append({})

                        if tool_call.id:
                            tool_calls_accumulated[index]['id'] = tool_calls_accumulated[index].get('id', '') + tool_call.id
                        if tool_call.function and tool_call.function.name:
                            tool_calls_accumulated[index]['name'] = tool_calls_accumulated[index].get('name', '') + tool_call.function.name
                        if tool_call.function and tool_call.function.arguments:
                            tool_calls_accumulated[index]['arguments'] = tool_calls_accumulated[index].get('arguments', '') + tool_call.function.arguments

        # 构建 LangChain 格式的 tool_calls
        lc_tool_calls = []
        for tc in tool_calls_accumulated:
            lc_tool_calls.append({
                "name": tc['name'],
                "args": json.loads(tc['arguments']) if tc['arguments'] else {},
                "id": tc['id']
            })
        
        return {"messages": [AIMessage(content=answer_content, tool_calls=lc_tool_calls)]}

    async def call_tools(self, state: AgentState):
        """执行工具调用节点"""
        last_message = state['messages'][-1]
        tool_calls = last_message.tool_calls
        
        console.print("\n[bold cyan]==== 工具调用 ====[/bold cyan]")
        results = []
        
        for tc in tool_calls:
            tool_name = tc['name']
            tool_args = tc['args']
            tool_id = tc['id']
            
            result = await self.session.call_tool(tool_name, tool_args)
            tool_result = result.content[0].text if result.content else "无结果"
            
            console.print(f"[bold yellow]工具名称:[/bold yellow] {tool_name}")
            console.print(f"[bold magenta]调用参数:[/bold magenta] {json.dumps(tool_args, ensure_ascii=False, indent=2)}")
            console.print(f"[bold green]返回结果:[/bold green] {tool_result}\n")
            
            results.append(ToolMessage(content=tool_result, tool_call_id=tool_id))
            
        return {"messages": results}

    async def process_query(self, query: str) -> str:
        """使用 LangGraph 处理查询并自动调用 MCP 工具。

        Args:
            query (str): 用户输入的自然语言查询。

        Returns:
            str: 模型的最终自然语言回复。
        """
        system_prompt = """你是一个很有帮助的助手。
                对于能够调用工具函数并行解决的问题，你可以一次性给出需要的函数调用。
                如果你认为某个问题可以被你很轻松地解决，不需要经过工具调用，你也可以选择直接跳过。
                如果你遇到了明显需要分步调用工具函数的问题，请你逐步来，不必一口气解决。
                在这种情况下，你会被多次调用，你的上一次对工具的调用会被记录，然后输入给你的下一次调用，
                直到不需要调用工具，最终再来输出回答。
                这份工具调用将严格按照时间顺序来记录，方便你分析前后顺序。
                请以友好的语气回答问题。"""
        
        inputs = {"messages": [SystemMessage(content=system_prompt), HumanMessage(content=query)]}
        
        # 执行图
        final_state = await self.app.ainvoke(inputs)
        
        return final_state['messages'][-1].content

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
