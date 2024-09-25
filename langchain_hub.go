package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/playwright-community/playwright-go"
)

func main() {
	// Initialize Playwright
	pw, err := playwright.Run()
	if err != nil {
		log.Fatalf("could not start Playwright: %v", err)
	}
	defer pw.Stop()

	// Launch a browser
	browser, err := pw.Chromium.Launch(playwright.BrowserTypeLaunchOptions{Headless: playwright.Bool(true)})
	if err != nil {
		log.Fatalf("could not launch Chromium: %v", err)
	}
	defer browser.Close()

	// Open a new page
	page, err := browser.NewPage()
	if err != nil {
		log.Fatalf("could not create new page: %v", err)
	}
	defer page.Close()

	// Go to the target webpage
	if _, err := page.Goto("https://smith.langchain.com/hub", playwright.PageGotoOptions{}); err != nil {
		log.Fatalf("could not goto page: %v", err)
	}

	// Wait for the page to load (similar to asyncio.sleep)
	time.Sleep(10 * time.Second)

	// Get the final page number using the relative XPath
	buttons, err := page.Locator(`//button[@class = "MuiButton-root MuiButton-variantOutlined MuiButton-colorNeutral MuiButton-sizeMd css-1bj1i78"]`).All()
	if err != nil {
		log.Fatalf("could not get total pages: %v", err)
	}

	// Check that the button array is not empty
	if len(buttons) == 0 {
		log.Fatalf("no pagination buttons found")
	}

	// Get the last button which contains the total page number
	finalPageNumber, err := buttons[len(buttons)-1].InnerText()
	if err != nil {
		log.Fatalf("could not get final page number: %v", err)
	}
	fmt.Printf("Final page number: %s\n", finalPageNumber)

	// Parse the final page number as an integer
	totalPages, err := strconv.Atoi(finalPageNumber)
	if err != nil {
		log.Fatalf("could not convert page number: %v", err)
	}

	// Data storage
	data := make(map[string]string)

	// Use sync.WaitGroup to run multiple scraping goroutines concurrently
	var wg sync.WaitGroup
	var mu sync.Mutex // To protect shared access to data map

	// Create a buffered channel to limit to 25 concurrent goroutines
	semaphore := make(chan struct{}, 25)

	for i := 1; i <= totalPages; i++ {
		wg.Add(1)

		// Acquire a spot in the semaphore
		semaphore <- struct{}{}

		go func(pageNum int) {
			defer wg.Done()

			// Release the spot in the semaphore when done
			defer func() { <-semaphore }()

			// Create a new page for each request (parallel execution)
			page, err := browser.NewPage()
			if err != nil {
				log.Printf("could not create page for page %d: %v", pageNum, err)
				return
			}
			defer page.Close()

			// Navigate to the specific page
			url := fmt.Sprintf("https://smith.langchain.com/hub?page=%d", pageNum)
			if _, err := page.Goto(url, playwright.PageGotoOptions{}); err != nil {
				log.Printf("could not goto page %d: %v", pageNum, err)
				return
			}

			// Wait for 10 seconds to allow the page to load
			time.Sleep(10 * time.Second)

			// Extract prompt names and descriptions
			promptNames, err := page.Locator(`//h4[@class = "text-lg font-medium"]`).All()
			if err != nil {
				log.Printf("could not get prompt names for page %d: %v", pageNum, err)
				return
			}
			promptDescriptions, err := page.Locator(`//div[@class = "text-sm"]`).All()
			if err != nil {
				log.Printf("could not get prompt descriptions for page %d: %v", pageNum, err)
				return
			}

			// Check that promptNames and promptDescriptions have the same length
			if len(promptNames) != len(promptDescriptions) {
				log.Printf("mismatch in lengths for page %d: promptNames=%d, promptDescriptions=%d", pageNum, len(promptNames), len(promptDescriptions))
				return
			}

			// Lock data access and add the scraped data
			mu.Lock()
			for x := range promptNames {
				name, err := promptNames[x].InnerText()
				if err != nil {
					log.Printf("could not get name for page %d: %v", pageNum, err)
					continue
				}
				description, err := promptDescriptions[x].InnerText()
				if err != nil {
					log.Printf("could not get description for page %d: %v", pageNum, err)
					continue
				}
				data[name] = description
			}
			mu.Unlock()

		}(i) // Pass page number to the goroutine
	}

	// Wait for all goroutines to finish
	wg.Wait()

	// Save data to JSON
	file, err := os.Create("prompts.json")
	if err != nil {
		log.Fatalf("could not create JSON file: %v", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	if err := encoder.Encode(data); err != nil {
		log.Fatalf("could not encode data to JSON: %v", err)
	}

	fmt.Println("Scraping complete and data saved.")
}
