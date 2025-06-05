from agents import Agent, Runner, InputGuardrail, GuardrailFunctionOutput, trace, tool
from pydantic import BaseModel
import asyncio
import logging
# import requests # Unused import removed
from bs4 import BeautifulSoup
from typing import List, Optional, Dict, Set, Tuple, Union
import re
# from urllib.parse import urlparse # Duplicate import removed
import aiohttp
from asyncio import TimeoutError
from urllib.parse import urlparse, unquote, urljoin
from dataclasses import dataclass
from datetime import datetime
import json
# import os # No longer needed if OPENREGISTERS_API_KEY is removed
# from dotenv import load_dotenv # No longer needed

# # Load environment variables from .env file # No longer needed
# load_dotenv() # No longer needed

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define output models
class TeamMember(BaseModel):
    name: str
    role: Optional[str] = None
    contact: Optional[str] = None # Keep as string, but instruct agent to be thorough

class GeneralContactInfo(BaseModel):
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    contact_form_url: Optional[str] = None
    # Add other common fields if needed, e.g., fax, social_media_links (as Dict[str, str] or another Pydantic model)

class FinancialHighlightItem(BaseModel):
    metric: str # e.g., "Revenue 2023", "Total Funding"
    value: str  # e.g., "€10M", "$50M"
    source: Optional[str] = None # e.g., "Annual Report", "Crunchbase"

class CompanyInfo(BaseModel):
    name: str
    website: str
    employee_count: Optional[str]
    general_contact_info: Optional[GeneralContactInfo] = None # Changed from Dict to specific Pydantic model
    service_areas: List[str]
    team_members: List[TeamMember]
    ownership_structure: Optional[str] = None
    latest_news_summary: Optional[str] = None
    geographic_focus: Optional[List[str]] = None
    sub_sector: Optional[str] = None
    financial_highlights: Optional[List[FinancialHighlightItem]] = None # New structure

class WebsiteVerification(BaseModel):
    is_correct: bool
    confidence: float
    reasoning: str

class FoundWebsite(BaseModel):
    url: Optional[str] = None
    reasoning: Optional[str] = None
    confidence: Optional[float] = None # 0.0 to 1.0

class BusinessProfileData(BaseModel):
    geographic_focus: Optional[List[str]] = None
    sub_sector: Optional[str] = None
    financial_highlights: Optional[List[FinancialHighlightItem]] = None # New structure

# New models for structured website content
@dataclass
class PageContent:
    url: str
    title: str
    content: str
    metadata: Dict[str, str]
    tables: List[Dict[str, List[str]]]
    lists: List[List[str]]
    links: List[str]
    last_modified: Optional[datetime] = None

@dataclass
class WebsiteStructure:
    base_url: str
    pages: Dict[str, PageContent]  # url -> PageContent
    navigation: Dict[str, List[str]]  # section -> list of urls
    sitemap: Optional[List[str]] = None

class StructuredContent(BaseModel):
    """Model for structured content extracted from website pages"""
    page_url: str
    section: str  # e.g., "about", "team", "contact"
    content_type: str  # e.g., "text", "table", "list"
    content: Dict[str, Union[str, List[str], List[List[str]], Dict[str, str]]]  # More specific content types
    confidence: float  # confidence score for extraction
    source_element: str  # HTML element type where found
    context: Optional[Dict[str, str]] = None  # Additional context about where/how the content was found

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"
    }

# Define the agents
search_agent = Agent(
    name="Official Website Finder Agent",
    instructions="""You are a specialist in finding official company websites.
    Given a company name, you MUST use the available web search tool to find relevant web pages.
    Analyze the search results from the tool to identify the single most likely official website for the company.
    Prioritize direct company domains (e.g., companyname.com) over directories, social media, or news articles unless they are the only strong indicators.
    If you find a likely official website, provide its URL, your reasoning, and a confidence score (0.0 to 1.0).
    If you cannot confidently identify an official website from the search results, set the URL to null and explain why.""",
    tools=[tool.WebSearchTool()], # Equipped with the WebSearchTool
    output_type=FoundWebsite
)

verification_agent = Agent(
    name="Website Verification Agent",
    instructions="""You verify if a given website belongs to the specified company.
    The website content might be in English, German, or Dutch.
    Check the website content, company name, and other identifying information (like address, contact details, or legal mentions such as 'Impressum' or 'Handelsregister').
    Provide a confidence score and reasoning for your verification.""",
    output_type=WebsiteVerification
)

async def extract_structured_content(soup: BeautifulSoup, url: str, section: str) -> List[StructuredContent]:
    """Extract structured content from a BeautifulSoup object"""
    structured_content = []
    
    # Extract tables
    for table in soup.find_all('table'):
        rows = []
        headers = []
        # Get headers
        header_row = table.find('thead')
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
        
        # Get rows
        for row in table.find_all('tr'):
            cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
            if cells:
                rows.append(cells)
        
        if rows:
            # Check if this table contains performance indicators
            table_text = ' '.join([cell for row in rows for cell in row]).lower()
            performance_indicators = {
                'locations': ['locations', 'offices', 'branches', 'standorte', 'filialen'],
                'customers': ['customers', 'clients', 'kunden', 'klanten'],
                'projects': ['projects', 'cases', 'projekte', 'cases'],
                'employees': ['employees', 'team', 'staff', 'mitarbeiter', 'medewerkers']
            }
            
            context = {}
            for indicator, keywords in performance_indicators.items():
                if any(keyword in table_text for keyword in keywords):
                    context[indicator] = 'Found in table'
            
            structured_content.append(StructuredContent(
                page_url=url,
                section=section,
                content_type="table",
                content={
                    "headers": headers,
                    "rows": rows
                },
                confidence=0.9,
                source_element="table",
                context=context if context else None
            ))
    
    # Extract lists
    for list_elem in soup.find_all(['ul', 'ol']):
        items = [li.get_text(strip=True) for li in list_elem.find_all('li')]
        if items:
            # Check if this list contains performance indicators
            list_text = ' '.join(items).lower()
            context = {}
            
            # Check for location patterns
            location_patterns = [
                r'(\d+)\s+(?:locations|offices|branches|standorte|filialen)',
                r'(?:locations|offices|branches|standorte|filialen):\s*(\d+)',
                r'(\d+)\s+(?:cities|countries|länder|landen)'
            ]
            for pattern in location_patterns:
                if match := re.search(pattern, list_text, re.IGNORECASE):
                    context['locations'] = f"Found in list: {match.group(1)}"
                    break
            
            # Check for customer/project patterns
            customer_project_patterns = [
                r'(\d+)\s+(?:customers|clients|kunden|klanten)',
                r'(\d+)\s+(?:projects|cases|projekte)',
                r'(?:serving|served)\s+(\d+)\s+(?:customers|clients)',
                r'(\d+)\s+(?:successful|completed)\s+(?:projects|cases)'
            ]
            for pattern in customer_project_patterns:
                if match := re.search(pattern, list_text, re.IGNORECASE):
                    if 'customers' in pattern or 'clients' in pattern:
                        context['customers'] = f"Found in list: {match.group(1)}"
                    else:
                        context['projects'] = f"Found in list: {match.group(1)}"
                    break
            
            structured_content.append(StructuredContent(
                page_url=url,
                section=section,
                content_type="list",
                content={"items": items},
                confidence=0.9,
                source_element=list_elem.name,
                context=context if context else None
            ))
    
    # Extract key-value pairs and performance indicators
    key_value_patterns = [
        # Basic key: value
        (r'(\w+)[\s:]+([^.\n]+)', 'text'),
        
        # Employee count patterns
        (r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(employees|staff|team members|mitarbeiter|medewerkers)', 'employee_count'),
        
        # Location patterns
        (r'(\d+)\s+(?:locations|offices|branches|standorte|filialen)', 'location_count'),
        (r'(?:locations|offices|branches|standorte|filialen):\s*(\d+)', 'location_count'),
        (r'(\d+)\s+(?:cities|countries|länder|landen)', 'location_count'),
        
        # Customer/Project patterns
        (r'(\d+)\s+(?:customers|clients|kunden|klanten)', 'customer_count'),
        (r'(\d+)\s+(?:projects|cases|projekte)', 'project_count'),
        (r'(?:serving|served)\s+(\d+)\s+(?:customers|clients)', 'customer_count'),
        (r'(\d+)\s+(?:successful|completed)\s+(?:projects|cases)', 'project_count'),
        
        # Financial patterns
        (r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|mio|mrd|€|$)', 'financial_value'),
        (r'revenue|umsatz|omzet:\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|mio|mrd|€|$)', 'revenue'),
        (r'funding|investment|finanzierung|investering:\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|mio|mrd|€|$)', 'funding'),
        (r'profit|gewinn|winst:\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|mio|mrd|€|$)', 'profit'),
        
        # Other company details
        (r'founded in (\d{4})', 'founding_year'),
        (r'(\+\d{1,3}[-.\s]?\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4})', 'phone'),
        (r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', 'email'),
    ]
    
    text_content = soup.get_text()
    for pattern, content_type in key_value_patterns:
        matches = re.finditer(pattern, text_content, re.IGNORECASE)
        for match in matches:
            context = {}
            if content_type == 'text':
                key, value = match.groups()
                content = {
                    "key": key.strip(),
                    "value": value.strip()
                }
            elif content_type in ['employee_count', 'location_count', 'customer_count', 'project_count']:
                count, _ = match.groups()
                content = {
                    "count": count.replace(',', '')
                }
                context[content_type] = f"Found in text: {count}"
            elif content_type in ['financial_value', 'revenue', 'funding', 'profit']:
                value = match.group(1)
                unit = match.group(2) if len(match.groups()) > 1 else ''
                content = {
                    "value": value.replace(',', ''),
                    "unit": unit
                }
                context[content_type] = f"Found in text: {value} {unit}"
            else:
                content = {
                    "value": match.group(1)
                }
            
            structured_content.append(StructuredContent(
                page_url=url,
                section=section,
                content_type=content_type,
                content=content,
                confidence=0.8,
                source_element="text",
                context=context if context else None
            ))
    
    return structured_content

async def crawl_website(base_url: str, max_pages: int = 10) -> WebsiteStructure:
    """Crawl website and extract structured content"""
    visited_urls: Set[str] = set()
    pages: Dict[str, PageContent] = {}
    navigation: Dict[str, List[str]] = {
        "about": [],
        "team": [],
        "contact": [],
        "services": [],
        "careers": [],
        "news": [],
        "legal": []
    }
    
    async def process_page(url: str, section: str = "main") -> None:
        if len(visited_urls) >= max_pages or url in visited_urls:
            return
        
        visited_urls.add(url)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=20) as response:
                    if response.status != 200:
                        return
                    
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Extract page metadata
                    metadata = {
                        "title": soup.title.string if soup.title else "",
                        "description": soup.find("meta", {"name": "description"})["content"] if soup.find("meta", {"name": "description"}) else "",
                        "keywords": soup.find("meta", {"name": "keywords"})["content"] if soup.find("meta", {"name": "keywords"}) else ""
                    }
                    
                    # Extract structured content
                    structured_content = await extract_structured_content(soup, url, section)
                    
                    # Extract tables and lists
                    tables = []
                    for table in soup.find_all('table'):
                        rows = []
                        for row in table.find_all('tr'):
                            cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                            if cells:
                                rows.append(cells)
                        if rows:
                            tables.append(rows)
                    
                    lists = []
                    for list_elem in soup.find_all(['ul', 'ol']):
                        items = [li.get_text(strip=True) for li in list_elem.find_all('li')]
                        if items:
                            lists.append(items)
                    
                    # Extract links
                    links = []
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if href.startswith('/'):
                            href = urljoin(base_url, href)
                        if href.startswith(base_url):
                            links.append(href)
                    
                    # Create PageContent
                    page_content = PageContent(
                        url=url,
                        title=metadata["title"],
                        content=soup.get_text(separator=' ', strip=True),
                        metadata=metadata,
                        tables=tables,
                        lists=lists,
                        links=links
                    )
                    
                    pages[url] = page_content
                    
                    # Categorize page
                    for section_name, patterns in {
                        "about": ["about", "über", "over"],
                        "team": ["team", "management", "leadership"],
                        "contact": ["contact", "kontakt"],
                        "services": ["services", "products", "solutions"],
                        "careers": ["careers", "jobs", "karriere"],
                        "news": ["news", "press", "blog"],
                        "legal": ["legal", "impressum", "privacy"]
                    }.items():
                        if any(pattern in url.lower() or pattern in metadata["title"].lower() for pattern in patterns):
                            navigation[section_name].append(url)
                    
                    # Process linked pages
                    for link in links:
                        if link not in visited_urls:
                            await process_page(link)
        
        except Exception as e:
            logger.error(f"Error processing page {url}: {str(e)}")
    
    await process_page(base_url)
    return WebsiteStructure(base_url=base_url, pages=pages, navigation=navigation)

# Update the scraping agent to use the new crawling functionality
scraping_agent = Agent(
    name="Web Scraping Agent",
    instructions="""You are a web scraping specialist. The website content may be in English, German, or Dutch.
    Your task is to thoroughly explore the company's website and its subpages to extract comprehensive information.
    
    You will receive structured content from multiple pages, including:
    - Raw text content
    - Extracted tables and lists
    - Key-value pairs
    - Page metadata
    - Navigation structure
    - Performance indicators
    - Financial information
    
    For each type of content, follow these guidelines:
    
    1. Tables and Lists:
       - Look for employee counts in tables with headers like "Company Size", "Team", "About Us"
       - Check lists for team members, services, or locations
       - Pay attention to structured data in tables (e.g., contact information, company details)
       - Look for performance indicators in tables (locations, customers, projects)
    
    2. Key-Value Pairs:
       - Extract employee counts from patterns like "X employees", "Team of X"
       - Look for contact information in key-value formats
       - Identify company details like founding year, legal form
       - Extract performance indicators (locations, customers, projects)
       - Look for financial information (revenue, funding, profit)
    
    3. Raw Text:
       - Search for employee count mentions in context
       - Look for team member information in paragraphs
       - Extract service descriptions and company details
       - Find performance indicators in text
       - Look for financial information in context
    
    4. Page Structure:
       - Use the navigation structure to understand the website organization
       - Pay special attention to About, Team, and Contact pages
       - Check footer and header sections for important information
       - Look for financial reports or investor relations pages
    
    5. Performance Indicators:
       - Number of locations/offices/branches
       - Customer/client count
       - Number of completed projects/cases
       - Employee count and growth
       - Market presence indicators
    
    6. Financial Information:
       - Revenue figures (look for terms like "revenue", "umsatz", "omzet")
       - Funding amounts (look for "funding", "investment", "finanzierung")
       - Profit figures (look for "profit", "gewinn", "winst")
       - Growth metrics
       - Financial year information
    
    Important Notes:
    - Cross-reference information across different pages
    - Verify numbers from multiple sources
    - Look for both current and historical information
    - Consider language-specific terms and formats
    - Prioritize structured data over raw text when available
    - Pay attention to the context of financial information
    - Look for official financial reports or investor relations pages
    
    Format the information clearly and verify its accuracy. If certain information is not found, indicate that it's not available rather than making assumptions.""",
    output_type=CompanyInfo
)

ownership_research_agent = Agent(
    name="Ownership Research Agent",
    instructions="""You are a specialist in determining company ownership structures. The company website or related documents might be in English, German, or Dutch.
    Given a company name and potentially its website, find out if the company is publicly traded, privately held, a subsidiary of another company, government-owned, or a non-profit. 
    Provide a concise description of its ownership structure. Look for clues in sections like "About Us", "Über uns", "Over ons", "Investor Relations", or legal notices like "Impressum". 
    If it's a subsidiary, try to name the parent company. Consider terms like 'AG', 'GmbH' (German), 'BV', 'NV' (Dutch) which indicate legal forms.
    Use the provided website content and also perform web searches if necessary for confirmation or details not on the site.""",
    output_type=str  # The output will be a string describing the ownership
)

news_research_agent = Agent(
    name="Company News Research Agent",
    instructions="""You are a specialist in finding and summarizing recent news about companies.
    Given a company name, use the web search tool to find 2-3 recent (within the last 6-12 months if possible) news articles or significant mentions about the company.
    Summarize the key findings from these news items in a concise paragraph. Focus on significant events like funding rounds, major product launches, partnerships, executive changes, or market performance.
    If no significant recent news is found, state that.""",
    tools=[tool.WebSearchTool()],
    output_type=str # The output will be a string (news summary)
)

key_person_research_agent = Agent(
    name="Key Person Research Agent",
    instructions="""You are a specialist in researching key individuals (executives, leaders, key team members) associated with a given company.
    Your goal is to identify these individuals and find their publicly available professional contact information.
    
    Inputs (you will be given):
    - Company Name.
    - Optionally, a list of already known team members (names/roles) that might have been found from the company's main website.

    Process:
    1.  If a list of known team members is provided, your primary goal is to verify their roles and find richer contact information for them.
    2.  If no team members are provided, or if you need to find additional key personnel (e.g., C-suite, VPs, Heads of Departments), identify them first.
    3.  For each key individual (either provided or newly identified by you):
        a.  Confirm their current role at the company if possible.
        b.  Use the WebSearchTool extensively to find their contact details. Formulate targeted queries such as:
            - '[Full Name] [Company Name] email address'
            - '[Full Name] [Company Name] contact information'
            - '[Full Name] [Company Name] LinkedIn profile'
            - 'site:linkedin.com/in/ [Full Name] [Potential Title] [Company Name]' (if you can infer a title)
            - '[Role, e.g., CEO] [Company Name] contact'
        c.  From the search results (snippets and potentially titles/URLs), extract:
            - Professional email addresses.
            - LinkedIn profile URLs.
            - Other relevant professional contact points (e.g., work phone if explicitly public, links to personal professional blogs if they contain contact info).
        d.  Prioritize official company pages, reputable business news/directories, and professional networking sites (like LinkedIn results from search).
        e.  Be cautious about outdated information. Note if information seems old.
    
    Output:
    Return a list of team members. Each team member should be an object with the following fields:
    - 'name': (Full Name)
    - 'role': (Their title/role at the company)
    - 'contact': (A string containing the found contact details. If multiple points are found, concatenate them clearly, e.g., "Email: jane.doe@example.com, LinkedIn: linkedin.com/in/janedoe, Phone: +1-555-1234 (publicly listed)"). If no specific contact info is found beyond a LinkedIn profile, list that.
    
    If enriching an existing list, update the contact details or roles if more accurate/detailed information is found. If no new contact info is found for a known person, preserve their existing details if any.
    If you cannot find any key individuals or their contact information despite searching, return an empty list or a list with individuals but with 'contact' indicating 'Not found'.""",
    tools=[tool.WebSearchTool()],
    output_type=List[TeamMember] 
)

business_profile_agent = Agent(
    name="Business Profile Agent",
    instructions="""You are a specialist in profiling companies beyond basic scraping.
    Given a company name, its website content, and potentially recent news, your task is to determine:
    1.  Geographic Focus: Identify the primary countries, regions, or markets the company operates in or targets (e.g., "USA & Canada", "DACH Region", "Global", "Southeast Asia"). Look for office locations, language of services, customer case studies, or direct mentions.
    2.  Sub-sector: Define the specific niche or sub-sector of its industry (e.g., "Cybersecurity for IoT devices", "AI-powered logistics optimization", "Sustainable fashion e-commerce platform"). This should be more specific than broad industry categories.
    3.  Verifiable Financial Highlights: Search for any publicly available and verifiable financial information. This could include:
        - Recent revenue figures (specify year and source if possible, e.g., "€15M in 2023 (Annual Report)").
        - Total funding raised (specify source, e.g., "$25M (Crunchbase)").
        - Profitability mentions if from a reputable source (e.g., "Reached profitability in Q4 2023 (CEO Statement in Forbes)").
        - Employee count changes or significant growth metrics if reported with a source.
    Use the WebSearchTool to find this information. Prioritize official company reports, reputable financial news outlets (e.g., Bloomberg, Reuters, Forbes, TechCrunch for funding), or well-known business data platforms (recognizing that direct access might be limited, but snippets from search results can be used).
    For financial data, always try to indicate the source or basis of the information.
    If specific information for any of these points cannot be found or verified, indicate 'Not found' or 'Not publicly available'.
    Return the findings structured with 'geographic_focus' (list of strings), 'sub_sector' (string), and 'financial_highlights' (a list of objects, where each object has 'metric', 'value', and 'source' fields).""",
    tools=[tool.WebSearchTool()],
    output_type=BusinessProfileData
)

# --- OpenRegisters API Integration --- # Removed entire section
# class OpenRegistersAutocompleteItem(BaseModel):
#     name: str
#     openregisters_id: str
#     jurisdiction: Optional[str] = None
#     legal_form: Optional[str] = None
#
# # Helper function to call OpenRegisters Autocomplete API
# async def autocomplete_openregisters_api(api_key: str, company_name: str) -> Optional[List[OpenRegistersAutocompleteItem]]:
#     # ... implementation removed ...
#
# # Helper function to call OpenRegisters Details API (this function remains largely the same)
# async def get_openregisters_details_api(api_key: str, openregisters_id: str) -> Optional[Dict]:
#     # ... implementation removed ...
#
# registry_research_agent = Agent(
#     name="Official Registry Research Agent",
#     # ... instructions removed ...
#     output_type=str # Outputs a string summary
# )
# --- End OpenRegisters API Integration --- 

# Add back the get_website_content function
async def get_website_content(url: str) -> str:
    """Get content from a website"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=headers, timeout=20, allow_redirects=True) as response: 
                    if response.status != 200:
                        raise Exception(f"Website request failed with status {response.status} for URL {response.url}")
                    content = await response.text()
            except TimeoutError:
                logger.error(f"Website request timed out for {url}")
                raise 
            except Exception as e:
                logger.error(f"Website request failed for {url}: {str(e)}")
                raise 
        
        soup = BeautifulSoup(content, 'html.parser')
        
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'iframe']): 
            element.decompose()
        
        text = soup.get_text(separator=' ', strip=True) 
        
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk and len(chunk) > 1) 
        
        return text
            
    except Exception as e:
        logger.error(f"Error getting website content for {url}: {e}", exc_info=True)
        return "" 

# Define guardrails with tracing
async def website_verification_guardrail(input_data: str, context: dict) -> GuardrailFunctionOutput:
    """Guardrail to ensure website verification is performed before scraping"""
    try:
        if not context.get('website_verified', False):
            return GuardrailFunctionOutput(
                output_info={"is_verified": False},
                tripwire_triggered=True
            )
        return GuardrailFunctionOutput(
            output_info={"is_verified": True},
            tripwire_triggered=False
        )
    except Exception as e:
        logger.error(f"Error in website verification guardrail: {e}", exc_info=True)
        return GuardrailFunctionOutput(
            output_info={"is_verified": False},
            tripwire_triggered=True
        )

# Main research agent with guardrails
main_agent = Agent(
    name="Company Research Agent",
    instructions="""You are a company research agent that helps users gather information about companies.
    You coordinate the research process, ensuring accurate website verification and data extraction.""",
    input_guardrails=[
        InputGuardrail(guardrail_function=website_verification_guardrail)
    ],
    handoffs=[search_agent, verification_agent, scraping_agent, ownership_research_agent, news_research_agent, key_person_research_agent, business_profile_agent]
)

async def chat_interface():
    """Interactive chat interface for the company research agent"""
    print("Welcome to the Company Research Agent!")
    print("=" * 50)
    print("I'll help you gather information about companies from their websites.")
    
    # openregisters_api_key = os.getenv("OPENREGISTERS_API_KEY") # Removed
    # if not openregisters_api_key: # Removed
    #     logger.warning("OPENREGISTERS_API_KEY not found in .env file. Official registry lookup will be skipped.") # Removed
    #     print("\nWARNING: OPENREGISTERS_API_KEY not set. Official registry data will not be fetched.\n") # Removed

    with trace("Company Research Session Workflow") as session_workflow_trace: 
        context = {'website_verified': False}
        
        while True:
            website_url = "" 
            verification_result = None 
            website_content = ""
            company_name = ""
            
            try:
                current_company_name_input = input("\nEnter company name (or 'quit' to exit): ")
                
                if current_company_name_input.lower() == 'quit':
                    print("\nThank you for using the Company Research Agent!")
                    break 
                
                company_name = current_company_name_input
                logger.info(f"[Session: {session_workflow_trace.trace_id if session_workflow_trace else 'N/A'}] Starting research for company: {company_name}")

                # All subsequent Runner.run calls will be spans under "Company Research Session Workflow"
                # Their inputs (including company_name) will differentiate them in the trace.

                # Search for company website
                print("\nSearching for company website...")
                try:
                    search_agent_result_obj = await Runner.run(
                        search_agent, 
                        f"Find the official website for the company: {company_name}",
                    ) 
                    
                    found_website_info = search_agent_result_obj.final_output

                    if not found_website_info or not found_website_info.url:
                        no_url_reason = found_website_info.reasoning if found_website_info and found_website_info.reasoning else "No suitable website could be identified."
                        print(f"Web search for '{company_name}' did not yield a confident official website. Reason: {no_url_reason}")
                        logger.info(f"Search agent could not identify official website for {company_name}. Reasoning: {no_url_reason}")
                        continue
                    
                    website_url = found_website_info.url
                    logger.info(f"Search agent identified official website for {company_name}: {website_url} with confidence {found_website_info.confidence}. Reasoning: {found_website_info.reasoning}")

                except TimeoutError: # This timeout is from asyncio.wait_for, which is not used here anymore for direct Runner.run
                                     # Timeouts for agent runs would typically be part of agent/tool config or a global RunConfig.
                    print("The search operation took too long. Please try again.")
                    logger.warning(f"Search operation for {company_name} might have timed out (if Runner.run has internal timeout).")
                    continue
                except Exception as e:
                    logger.error(f"An error occurred during the search agent run for {company_name}: {str(e)}", exc_info=True)
                    print(f"An error occurred during the search. Please try again.")
                    continue
                
                if not website_url:
                    print("Could not determine a website URL for the company.")
                    logger.info(f"No website URL determined for {company_name} after search agent.")
                    continue

                print(f"\nI will investigate this website: {website_url}")
                
                # Verify website
                print("\nVerifying if this is the correct website...")
                try:
                    website_content = await asyncio.wait_for(
                        get_website_content(website_url),
                        timeout=20 # Increased timeout for fetching potentially large/slow pages
                    )
                    if not website_content:
                        print(f"Could not retrieve content from {website_url}. It might be inaccessible or empty.")
                        user_choice = input("Do you want to (s)kip this company, or (q)uit? [s/q]: ").lower()
                        if user_choice == 'q':
                            print("\nThank you for using the Company Research Agent!")
                            return 
                        else:
                            logger.info(f"Skipping company {company_name} after failed content retrieval of {website_url}.")
                            continue

                    verification_prompt = f"Based on the following content from {website_url}, is this the official website for the company '{company_name}'? Consider if the company name is prominently displayed, if the content aligns with a typical official company site (e.g., about us, products/services), and if there are any red flags suggesting it's not official. Content snippet (up to 1000 chars):\n\n{website_content[:1000]}"
                    verification_result = await Runner.run(
                        verification_agent,
                        verification_prompt,
                        context=context
                    )
                except TimeoutError:
                    print(f"The website content retrieval or verification for {website_url} took too long. Please try again.")
                    logger.warning(f"Verification phase timed out for {website_url} ({company_name})")
                    continue
                except Exception as e:
                    logger.error(f"An error occurred during verification for {website_url} ({company_name}): {str(e)}", exc_info=True)
                    print(f"An error occurred during verification for {website_url}. Please try again.")
                    continue
                
                if not verification_result or not hasattr(verification_result.final_output, 'is_correct'):
                    print(f"Verification agent did not return a valid response for {website_url}.")
                    logger.warning(f"Invalid verification agent response for {company_name}, URL {website_url}.")
                    continue

                if not verification_result.final_output.is_correct:
                    print(f"\nBased on my analysis, {website_url} does not seem to be the correct official website. Reasoning: {verification_result.final_output.reasoning}")
                    user_choice = input("Do you want to (s)kip this company, or (q)uit? [s/q]: ").lower()
                    if user_choice == 'q':
                        print("\nThank you for using the Company Research Agent!")
                        return 
                    else:
                        logger.info(f"Skipping company {company_name} after website verification failed for {website_url}.")
                        continue
                
                print(f"\nWebsite {website_url} verified as correct (Confidence: {verification_result.final_output.confidence:.2f})! Proceeding with data extraction...")
                context['website_verified'] = True
                logger.info(f"Website {website_url} verified for {company_name}.")
                
                # Crawl website and get structured content
                print("\nCrawling website and extracting structured content...")
                try:
                    website_structure = await crawl_website(website_url)
                    
                    # Prepare structured content for the scraping agent
                    structured_content = []
                    for url, page in website_structure.pages.items():
                        # Add raw text content
                        structured_content.append({
                            "url": url,
                            "title": page.title,
                            "content": page.content,
                            "metadata": page.metadata,
                            "tables": page.tables,
                            "lists": page.lists
                        })
                    
                    # Convert structured content to a format suitable for the agent
                    scraping_prompt = f"""Analyze the following structured content from {company_name}'s website ({website_url}):

{json.dumps(structured_content, indent=2)}

Extract comprehensive information about the company, including:
- Number of employees
- Contact information
- Service areas
- Team members
- Company structure
- Any other relevant information

Pay special attention to structured data in tables and lists, as they often contain the most accurate information."""
                    
                    scraping_run_result = await Runner.run(
                        scraping_agent,
                        scraping_prompt,
                        context=context
                    )
                    
                    scraping_result_data = scraping_run_result.final_output
                except Exception as e:
                    logger.error(f"An error occurred during structured content extraction for {company_name}: {str(e)}", exc_info=True)
                    print(f"An error occurred during structured content extraction. Please try again.")
                    continue
                
                initial_team_members = []
                if scraping_result_data and scraping_result_data.team_members:
                    initial_team_members = scraping_result_data.team_members

                print("\nPerforming focused research on key persons...")
                key_persons_prompt = f"For the company '{company_name}', find key executives or team members (e.g., CEO, CTO, Head of Product, key managers). "
                if initial_team_members:
                    known_members_str = "\nKnown team members to verify/enrich (if possible, find specific emails or LinkedIn profiles):\n"
                    for tm in initial_team_members:
                        known_members_str += f"- Name: {tm.name}, Role: {tm.role or 'Unknown'}\n"
                    key_persons_prompt += known_members_str
                else:
                    key_persons_prompt += "No team members were identified from the main website; please try to find some key personnel using web search."

                key_persons_run_result = await Runner.run(
                    key_person_research_agent,
                    key_persons_prompt
                )
                
                final_team_members = initial_team_members 
                if key_persons_run_result and key_persons_run_result.final_output:
                    found_key_persons = key_persons_run_result.final_output
                    if found_key_persons:
                        final_team_members = found_key_persons 
                        logger.info(f"Key person research agent found/updated {len(found_key_persons)} team members for {company_name}.")
                    elif not initial_team_members:
                        logger.info(f"Key person research agent did not find any team members for {company_name}.")
                    else:
                         logger.info(f"Key person research agent did not add new key persons beyond the initial list for {company_name}.")
                else:
                    logger.info(f"No additional key person information found by the dedicated agent for {company_name}.")

                if scraping_result_data:
                    scraping_result_data.team_members = final_team_members
                elif final_team_members:
                    logger.warning("Scraping_result_data was None, but key persons were found. Display might be incomplete. Re-creating CompanyInfo for display.")
                    # If scraping_result_data is None, but we have team members, we need to create a CompanyInfo to hold them for display
                    # This assumes website_url and company_name are available from the broader scope
                    scraping_result_data = CompanyInfo(
                        name=company_name, 
                        website=website_url, 
                        team_members=final_team_members,
                        # Other fields will be None or empty list by default
                        service_areas=[],
                        # general_contact_info, employee_count, ownership_structure, latest_news_summary will be default (None)
                    )

                # Research Ownership Structure
                print("\nResearching ownership structure...")
                ownership_prompt = f"Determine the ownership structure of the company: {company_name}. Website for reference: {website_url}. Consider if it's public, private, a subsidiary (and of whom), non-profit, etc. Use the provided website content and also perform web searches if necessary for confirmation or details not on the site."
                if website_content:
                    ownership_prompt += f"\n\nPreviously scraped website content (for context, but perform your own checks):\n{website_content[:1500]}"
                
                ownership_run_result = await Runner.run(
                    ownership_research_agent, 
                    ownership_prompt
                )
                ownership_info_str = ownership_run_result.final_output if ownership_run_result and ownership_run_result.final_output else "Could not determine ownership structure."

                # Research Latest News
                news_summary_str = "Not researched yet."
                print("\nResearching latest news...")
                news_prompt = f"Find and summarize recent (last 6-12 months) news about the company: {company_name}. Focus on 2-3 key events like funding, product launches, partnerships, or executive changes."
                news_result_obj = await Runner.run(
                    news_research_agent,
                    news_prompt
                )
                news_summary_str = news_result_obj.final_output if news_result_obj and news_result_obj.final_output else "No significant recent news found or could not be summarized."

                # Research Business Profile (Geo, Sub-sector, Financials)
                bp_data = None
                print("\nResearching business profile (geographic focus, sub-sector, financials)...")
                bp_prompt = f"For the company '{company_name}' (website: {website_url}), determine its geographic focus, specific sub-sector, and any verifiable financial highlights. Use the provided website content and news summary for context, and perform web searches for additional details."
                if website_content:
                    bp_prompt += f"\n\nWebsite Content Snippet:\n{website_content[:1000]}..."
                if news_summary_str and news_summary_str != "No significant recent news found or could not be summarized.":
                    bp_prompt += f"\n\nRecent News Summary:\n{news_summary_str}"
                
                bp_run_result = await Runner.run(
                    business_profile_agent,
                    bp_prompt
                )
                
                if bp_run_result and bp_run_result.final_output:
                    bp_data = bp_run_result.final_output
                    # Update the main CompanyInfo object (scraping_result_data) with these new findings
                    if scraping_result_data: # Ensure scraping_result_data exists
                        scraping_result_data.geographic_focus = bp_data.geographic_focus
                        scraping_result_data.sub_sector = bp_data.sub_sector
                        scraping_result_data.financial_highlights = bp_data.financial_highlights # Assign the list of FinancialHighlightItem
                    elif company_name and website_url: # If scraping_result_data was None, create it
                        logger.warning("scraping_result_data was None, creating new CompanyInfo for business profile data.")
                        scraping_result_data = CompanyInfo(
                            name=company_name,
                            website=website_url,
                            team_members=[], # Will be empty if scraping agent failed to produce CompanyInfo
                            service_areas=[],
                            geographic_focus=bp_data.geographic_focus,
                            sub_sector=bp_data.sub_sector,
                            financial_highlights=bp_data.financial_highlights
                        )
                    logger.info(f"Business profile for {company_name}: Geo: {bp_data.geographic_focus}, Sub-sector: {bp_data.sub_sector}, Financials: {bp_data.financial_highlights}")
                else:
                    logger.info(f"Could not determine detailed business profile for {company_name}.")

                # Display results
                print("\n" + "=" * 50)
                print("COMPANY INFORMATION")
                print("=" * 50)
                if scraping_result_data:
                    final_output = scraping_result_data # Use the CompanyInfo object from scraping_agent
                    print(f"Company: {final_output.name if final_output.name else company_name}")
                    print(f"Website: {final_output.website if final_output.website else website_url}")
                    print(f"\nNumber of Employees: {final_output.employee_count if final_output.employee_count else 'Not found'}")
                    print(f"\nOwnership Structure: {ownership_info_str}") 

                    print("\nGeneral Contact Information:")
                    if final_output.general_contact_info:
                        contact_details = final_output.general_contact_info
                        if contact_details.phone: print(f"  Phone: {contact_details.phone}")
                        if contact_details.email: print(f"  Email: {contact_details.email}")
                        if contact_details.address: print(f"  Address: {contact_details.address}")
                        if contact_details.contact_form_url: print(f"  Contact Form: {contact_details.contact_form_url}")
                        if not any([contact_details.phone, contact_details.email, contact_details.address, contact_details.contact_form_url]):
                            print("  No specific details found within general contact info.")
                    else:
                        print("  Not found on main website.")

                    print("\nService Areas:")
                    if final_output.service_areas:
                        for area in final_output.service_areas:
                            print(f"- {area}")
                    else:
                        print("Not found or not applicable.")
                    
                    print("\nKey Team Members:")
                    if final_output.team_members:
                        for member in final_output.team_members:
                            print(f"\n- Name: {member.name if member.name else 'N/A'}")
                            print(f"  Role: {member.role if member.role else 'N/A'}")
                            print(f"  Contact: {member.contact if member.contact else 'N/A'}")
                    else:
                        print("Not found or not applicable.")
                else:
                    print("Could not extract detailed company information from scraping.")
                
                print("\nLatest News Summary:")
                print(news_summary_str) 

                print("\nGeographic Focus:")
                if scraping_result_data and scraping_result_data.geographic_focus:
                    print(", ".join(scraping_result_data.geographic_focus))
                else:
                    print("  Not found or not specified.") 
                
                print("\nSub-sector:")
                if scraping_result_data and scraping_result_data.sub_sector:
                    print(f"  {scraping_result_data.sub_sector}")
                else:
                    print("  Not found or not specified.")
                
                print("\nFinancial Highlights:")
                if scraping_result_data and scraping_result_data.financial_highlights:
                    for item in scraping_result_data.financial_highlights:
                        source_info = f" (Source: {item.source})" if item.source else ""
                        print(f"  - {item.metric}: {item.value}{source_info}")
                else:
                    print("  Not found or not publicly available.")

                print("\n" + "=" * 50)
                logger.info(f"Successfully processed company: {company_name}")
            
            except Exception as e:
                # This will catch errors from Runner.run if they are not caught by inner try-excepts,
                # or other unexpected errors within the loop for a specific company.
                logger.error(f"[SessionWorkflow] Unexpected error during processing for '{company_name}': {e}", exc_info=True)
                print(f"\nAn unexpected error occurred while processing '{company_name}': {e}. Please try again.")
            
            finally:
                context['website_verified'] = False
            
            continue_research = input("\nWould you like to research another company? (yes/no): ").lower()
            if continue_research.lower() != 'yes':
                print("\nThank you for using the Company Research Agent!")
                break # This will exit the loop and the session_workflow_trace context manager will finish the trace.

if __name__ == "__main__":
    asyncio.run(chat_interface())
