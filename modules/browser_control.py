import asyncio
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from typing import Dict, List, Optional, Union
import re

# Create a singleton instance
_controller = None

async def initialize():
    """Initialize the browser controller"""
    global _controller
    if not _controller:
        _controller = BrowserController()
        await _controller.initialize()
    return _controller

async def cleanup():
    """Clean up browser resources"""
    global _controller
    if _controller:
        await _controller.cleanup()
        _controller = None

async def search(query: str) -> Dict:
    """Perform web search"""
    global _controller
    if not _controller:
        await initialize()
    return await _controller.search(query)

async def check_email(provider: str) -> Dict:
    """Check email using specified provider"""
    global _controller
    if not _controller:
        await initialize()
    return await _controller.check_email(provider)

async def compose_email(to: str, subject: str, body: str, provider: str) -> Dict:
    """Compose email using specified provider"""
    global _controller
    if not _controller:
        await initialize()
    return await _controller.compose_email(to, subject, body, provider)

class BrowserController:
    """Browser automation controller using Playwright"""
    
    def __init__(self):
        self.browser = None
        self.context = None
        self.page = None
    
    async def initialize(self):
        """Initialize browser instance"""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(
            headless=False,
            args=['--start-maximized']
        )
        self.context = await self.browser.new_context(
            viewport={'width': 900, 'height': 600},
            accept_downloads=True
        )
        self.page = await self.context.new_page()
        self.page.set_default_timeout(30000)
    
    async def cleanup(self):
        """Clean up browser resources"""
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
    
    async def wait_for_navigation_complete(self):
        """Wait for any ongoing navigation to complete"""
        try:
            await self.page.wait_for_load_state('domcontentloaded', timeout=5000)
            await self.page.wait_for_load_state('networkidle', timeout=5000)
        except:
            pass
    
    async def navigate(self, url: str, timeout: Optional[int] = 30000) -> Dict:
        """Navigate to specified URL"""
        try:
            await self.wait_for_navigation_complete()
            
            if timeout and timeout < 1000:
                await asyncio.sleep(timeout / 1000)
                return {
                    "status": "error",
                    "message": f"Navigation timeout of {timeout}ms exceeded"
                }
            
            response = await self.page.goto(url, timeout=timeout, wait_until='domcontentloaded')
            if not response:
                return {
                    "status": "error",
                    "message": "Navigation failed"
                }
            
            await asyncio.sleep(1)
            
            return {
                "status": "success",
                "title": await self.page.title()
            }
            
        except PlaywrightTimeoutError as e:
            return {
                "status": "error",
                "message": str(e)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def check_for_captcha(self) -> bool:
        """Check if a CAPTCHA is present on the page"""
        captcha_selectors = [
            'iframe[src*="recaptcha"]',
            'iframe[src*="captcha"]',
            'div.g-recaptcha',
            '#captcha',
            'form#captcha',
            'div[aria-label*="CAPTCHA"]',
            'div[aria-label*="captcha"]',
            'img[alt*="CAPTCHA"]',
            'img[alt*="captcha"]'
        ]
        
        for selector in captcha_selectors:
            try:
                element = await self.page.wait_for_selector(selector, timeout=2000)
                if element:
                    return True
            except:
                continue
        
        content = await self.page.content()
        captcha_texts = ['captcha', 'CAPTCHA', 'verify you are human', 'security check']
        return any(text.lower() in content.lower() for text in captcha_texts)
    
    async def search(self, query: str) -> Dict:
        """Perform web search"""
        try:
            print("\nStarting search operation...")
            
            # Check if we're already on Google
            current_url = self.page.url
            if not re.search(r'google\.com', current_url, re.IGNORECASE):
                print("Navigating to Google...")
                nav_result = await self.navigate("https://www.google.com")
                if nav_result["status"] == "error":
                    print(f"Navigation error: {nav_result['message']}")
                    return nav_result
            
            # Wait for page to be ready
            await self.wait_for_navigation_complete()
            await asyncio.sleep(1)
            
            # Check for CAPTCHA
            if await self.check_for_captcha():
                return {
                    "status": "error",
                    "message": "CAPTCHA detected. Manual intervention required."
                }
            
            # Find and fill search input
            print("Looking for search input...")
            search_input = await self.page.wait_for_selector('textarea[name="q"]', timeout=5000)
            if not search_input:
                return {
                    "status": "error",
                    "message": "Could not find search input"
                }
            
            # Clear existing search input and enter new query
            print(f"Entering search query: {query}")
            await search_input.fill("")
            await asyncio.sleep(0.5)
            await search_input.fill(query)
            await search_input.press('Enter')
            
            # Wait for search results
            print("Waiting for search results...")
            await self.wait_for_navigation_complete()
            await asyncio.sleep(3)  # Give more time for results to load
            
            # Check for CAPTCHA again
            if await self.check_for_captcha():
                return {
                    "status": "error",
                    "message": "CAPTCHA detected after search. Manual intervention required."
                }
            
            # Wait for and verify search results container
            print("Looking for search results...")
            await self.page.wait_for_selector('#search', timeout=10000)
            
            # Extract results using JavaScript
            results = await self.page.evaluate('''() => {
                const results = [];
                const containers = document.querySelectorAll('#search .g, #rso > div');
                
                for (const container of containers) {
                    try {
                        const h3 = container.querySelector('h3');
                        if (!h3) continue;
                        
                        let link = h3.closest('a');
                        if (!link) {
                            link = container.querySelector('a[ping], a[href]:not([href="#"])');
                        }
                        
                        if (!link || !link.href || link.href.includes('google.com')) continue;
                        
                        const descEl = container.querySelector('div[style*="line-clamp"], div.VwiC3b, div[role="text"]');
                        
                        results.push({
                            title: h3.innerText.trim(),
                            url: link.href,
                            description: descEl ? descEl.innerText.trim() : ''
                        });
                        
                        if (results.length >= 5) break;
                    } catch (e) {
                        console.error('Error processing result:', e);
                    }
                }
                
                return results;
            }''')
            
            if not results:
                print("No valid results could be extracted")
                return {
                    "status": "error",
                    "message": "No valid results found"
                }
            
            print(f"Successfully extracted {len(results)} valid results")
            return {
                "status": "success",
                "results": results
            }
            
        except Exception as e:
            print(f"Search error: {str(e)}")
            return {
                "status": "error",
                "message": f"Search error: {str(e)}"
            }
    
    async def check_email(self, provider: str) -> Dict:
        """Check email using specified provider"""
        try:
            if provider.lower() == "gmail":
                await self.navigate("https://mail.google.com")
                
                try:
                    await self.page.wait_for_selector(
                        'div[role="main"], input[type="email"]',
                        timeout=10000
                    )
                except:
                    pass
                
                if await self.page.query_selector('input[type="email"]'):
                    return {
                        "status": "error",
                        "message": "Login required",
                        "provider": provider
                    }
                
                emails = []
                try:
                    elements = await self.page.query_selector_all('tr.zA.zE')
                    
                    for element in elements[:5]:
                        sender_el = await element.query_selector('.yX')
                        subject_el = await element.query_selector('.y6')
                        preview_el = await element.query_selector('.y2')
                        
                        if sender_el and subject_el:
                            sender = await sender_el.inner_text()
                            subject = await subject_el.inner_text()
                            preview = ''
                            
                            if preview_el:
                                preview = await preview_el.inner_text()
                            
                            emails.append({
                                "sender": sender,
                                "subject": subject,
                                "preview": preview,
                                "unread": True
                            })
                except:
                    pass
                
                return {
                    "status": "success",
                    "provider": provider,
                    "emails": emails
                }
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported email provider: {provider}",
                    "provider": provider
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "provider": provider
            }
    
    async def compose_email(self, to: str, subject: str, body: str, 
                          provider: str) -> Dict:
        """Compose email using specified provider"""
        try:
            if provider.lower() == "gmail":
                await self.navigate("https://mail.google.com")
                
                if await self.page.query_selector('input[type="email"]'):
                    return {
                        "status": "error",
                        "message": "Login required"
                    }
                
                compose_button = await self.page.wait_for_selector(
                    'div[role="button"][gh="cm"]',
                    timeout=10000
                )
                await compose_button.click()
                
                await self.page.wait_for_selector(
                    'div[role="dialog"]',
                    timeout=10000
                )
                
                to_input = await self.page.wait_for_selector(
                    'input[name="to"]',
                    timeout=5000
                )
                await to_input.fill(to)
                
                subject_input = await self.page.wait_for_selector(
                    'input[name="subjectbox"]',
                    timeout=5000
                )
                await subject_input.fill(subject)
                
                body_input = await self.page.wait_for_selector(
                    'div[role="textbox"]',
                    timeout=5000
                )
                await body_input.fill(body)
                
                return {
                    "status": "success",
                    "message": "Email composed successfully"
                }
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported email provider: {provider}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
