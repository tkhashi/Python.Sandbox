from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio
from dotenv import load_dotenv

load_dotenv()


async def main():
    agent = Agent(
        task="""
        アジアクエストという会社について調べて、競合優位性をまとめて。
        """,
        llm=ChatOpenAI(model="gpt-4o"),
    )
    result = await agent.run()
    print(result)


asyncio.run(main())
