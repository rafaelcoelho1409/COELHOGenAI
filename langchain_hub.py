import asyncio
from playwright.async_api import async_playwright
import json

async def run(playwright):
    # Launch a browser
    browser = await playwright.chromium.launch(headless = True)
    # Open a new page
    page = await browser.new_page()
    # Go to the target webpage
    await page.goto("https://smith.langchain.com/hub")
    # Wait for 10 seconds after the page has loaded
    await asyncio.sleep(10)
    # Use relative XPath to find an element (example: all links `//a`)
    total_page_numbers = await page.locator('//button[@class = "MuiButton-root MuiButton-variantOutlined MuiButton-colorNeutral MuiButton-sizeMd css-1bj1i78"]').all()
    # Iterate over all found elements and print the href attribute
    final_page_number = await total_page_numbers[-1].inner_text()

    data = {}
    for i in range(1, int(final_page_number) + 1):
        await page.goto(f"https://smith.langchain.com/hub?page={i}")
        await asyncio.sleep(10)
        prompt_names = await page.locator('//h4[@class = "text-lg font-medium"]').all()
        prompt_descriptions = await page.locator('//div[@class = "text-sm"]').all()
        for x, y in zip(prompt_names, prompt_descriptions):
            data[await x.inner_text()] = await y.inner_text()
    # Close the browser
    await browser.close()
    with open("prompts.json", "w") as f:
        json.dump(data, f)

# Run the script
async def main():
    async with async_playwright() as playwright:
        await run(playwright)

# Start the asyncio event loop
asyncio.run(main())
