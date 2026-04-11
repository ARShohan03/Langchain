"""Web scraping module for fetching scholarship programs."""

import time
import json
from typing import List, Dict, Any
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


def fetch_programs() -> List[Dict[str, Any]]:
    """
    Fetch Erasmus Mundus programs from official website.
    
    Returns:
        List of program dictionaries with title, url, region, and metadata.
    """
    print("\n" + "="*70)
    print("🎓 ERASMUS MUNDUS PROGRAM FETCHER")
    print("="*70)
    
    chrome_options = Options()
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    # Remove headless to see what's happening
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )
    
    # Set longer timeouts
    driver.set_page_load_timeout(180)
    driver.set_script_timeout(180)

    try:
        print("\n🔗 Step 1: Connecting to Erasmus Mundus website...")
        print("   URL: https://www.eacea.ec.europa.eu/scholarships/erasmus-mundus-catalogue_en")
        
        driver.get("https://www.eacea.ec.europa.eu/scholarships/erasmus-mundus-catalogue_en")
        wait = WebDriverWait(driver, 180)

        print("⏳ Step 2: Loading page content (this may take 60-90 seconds)...")
        time.sleep(10)  # Initial load
        
        # Scroll multiple times to trigger lazy loading
        print("📜 Step 3: Scrolling to load all programs...")
        for i in range(10):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            print(f"   Scroll {i+1}/10...")
            time.sleep(2)

        # Step 1: Collect all unique URLs
        print("\n🔍 Step 4: Finding program links...")
        
        # Get all links
        all_links = driver.find_elements(By.TAG_NAME, "a")
        urls = []
        seen_urls = set()
        
        for link in all_links:
            try:
                href = link.get_attribute("href")
                if href and "erasmus-plus.ec.europa.eu/projects" in href and href not in seen_urls:
                    urls.append(href)
                    seen_urls.add(href)
            except Exception:
                continue

        print(f"✅ Found {len(urls)} unique program URLs")

        if not urls:
            print("\n❌ ERROR: No programs found on website")
            print("Possible causes:")
            print("  1. Website structure has changed")
            print("  2. No internet connection")
            print("  3. Website is blocking automation")
            print("  4. Page didn't load properly")
            raise Exception("Could not find any programs on website")

        programs = []

        # Step 2: Visit each URL and extract details
        print(f"\n📚 Step 5: Extracting details from {len(urls)} programs...")
        print("   (This may take several minutes)")
        
        for idx, url in enumerate(urls, 1):
            try:
                print(f"\n   [{idx}/{len(urls)}] Fetching: {url}")
                driver.get(url)
                time.sleep(4)  # Wait for page to load

                # Extract title
                try:
                    title = driver.find_element(By.TAG_NAME, "h1").text.strip()
                except Exception:
                    title = driver.find_element(By.TAG_NAME, "h2").text.strip() if driver.find_elements(By.TAG_NAME, "h2") else "Unknown Program"

                # Get page content
                try:
                    page_text = driver.find_element(By.TAG_NAME, "body").text[:1000]
                except Exception:
                    page_text = title

                programs.append({
                    "title": title,
                    "url": url,
                    "region": "Europe",
                    "countries": "",
                    "universities": "",
                    "degree": "Master's",
                    "duration": "2 years",
                    "text": f"{title}. Program information available at {url}",
                })
                print(f"       ✓ {title[:60]}")

            except Exception as e:
                print(f"       ⚠️  Error fetching this program: {str(e)[:50]}")
                continue

        if not programs:
            print("\n❌ ERROR: Could not extract any program details")
            raise Exception("Could not extract program details from website")

        print(f"\n{'='*70}")
        print(f"✅ SUCCESS: Fetched {len(programs)} programs from Erasmus Mundus website!")
        print(f"{'='*70}\n")
        
        return programs
    
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        print("\nDEBUG INFO:")
        print(f"  Current URL: {driver.current_url}")
        print(f"  Page title: {driver.title}")
        raise
    
    finally:
        print("\n🔒 Closing browser...")
        driver.quit()


def fetch_erasmus_programs() -> List[Dict[str, Any]]:
    """Alias for fetch_programs."""
    return fetch_programs()
