import asyncio
from browser_use_tool import BrowserUseTool

async def main():
    tool = BrowserUseTool()
    result = await tool.execute(action="go_to_url", url="https://www.google.lk/")
    print(result.output)
    state = await tool.get_current_state()
    print(state.output)
    await tool.cleanup()

asyncio.run(main())