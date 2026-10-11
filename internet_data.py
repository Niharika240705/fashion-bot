"""
Internet Data Fetching Module for Fashion Bot
Fetches real-time fashion data from Google, Wikipedia, and web sources
"""

import requests
from bs4 import BeautifulSoup
import wikipedia
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InternetDataFetcher:
    """
    Fetches real-time fashion data from internet sources.
    
    Features:
    - Google Search integration
    - Wikipedia fashion information
    - Web scraping for prices and products
    - Trend data fetching
    - Price comparison
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.cache = {}
        self.cache_time = {}
        
    def search_google(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search Google for fashion information.
        Uses SerpAPI fallback since Google doesn't directly support scraping.
        """
        try:
            # Method 1: Try using Wikipedia for fashion-related queries
            if 'fashion' in query.lower() or 'brand' in query.lower():
                return self._search_wikipedia(query, max_results)
            
            # Method 2: Generic web search
            results = self._web_search(query, max_results)
            return results
        
        except Exception as e:
            logger.error(f"Error searching Google: {e}")
            return []
    
    def _search_wikipedia(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search Wikipedia for fashion-related information."""
        try:
            # Search Wikipedia
            search_results = wikipedia.search(query, results=3)
            
            results = []
            for title in search_results[:max_results]:
                try:
                    page = wikipedia.page(title)
                    results.append({
                        'title': page.title,
                        'url': page.url,
                        'summary': page.summary[:300] + '...',
                        'source': 'Wikipedia',
                        'fetched_at': datetime.now().isoformat()
                    })
                except wikipedia.exceptions.DisambiguationError:
                    continue
        
            return results
        
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return []
    
    def _web_search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Generic web search using Google's search API.
        Falls back to basic URL construction if API unavailable.
        """
        try:
            # Build Google search URL
            search_url = f"https://www.google.com/search?q={query}"
            
            response = self.session.get(search_url, timeout=5)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            results = []
            
            # Extract search results
            for g in soup.find_all('div', class_='g')[:max_results]:
                try:
                    title_elem = g.find('h3')
                    link_elem = g.find('a')
                    desc_elem = g.find('div', class_='VwiC3b')
                    
                    if title_elem and link_elem:
                        results.append({
                            'title': title_elem.get_text(),
                            'url': link_elem.get('href'),
                            'summary': desc_elem.get_text() if desc_elem else 'No description',
                            'source': 'Google Search',
                            'fetched_at': datetime.now().isoformat()
                        })
                except:
                    continue
            
            return results
        
        except Exception as e:
            logger.error(f"Web search error: {e}")
            # Return fashion brand information as fallback
            return self._get_brand_info(query)
    
    def search_fashion_brand(self, brand_name: str) -> Dict:
        """Search for fashion brand information."""
        
        # Check cache first
        cache_key = f"brand_{brand_name.lower()}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        results = self.search_google(f"{brand_name} luxury fashion brand")
        
        info = {
            'brand': brand_name,
            'search_results': results,
            'local_info': self._get_brand_info(brand_name),
            'timestamp': datetime.now().isoformat()
        }
        
        self.cache[cache_key] = info
        return info
    
    def search_fashion_product(self, product_name: str, brand: Optional[str] = None) -> Dict:
        """Search for fashion product information and prices."""
        
        query = f"{brand} {product_name}" if brand else product_name
        cache_key = f"product_{query.lower()}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        results = self.search_google(f"{query} luxury fashion price")
        
        product_info = {
            'product': product_name,
            'brand': brand,
            'search_results': results,
            'estimated_price_range': self._estimate_price(product_name),
            'timestamp': datetime.now().isoformat()
        }
        
        self.cache[cache_key] = product_info
        return product_info
    
    def search_fashion_trends(self) -> Dict:
        """Search for current fashion trends."""
        
        if 'trends' in self.cache:
            return self.cache['trends']
        
        trends_query = "2026 fashion trends luxury brands"
        results = self.search_google(trends_query, max_results=10)
        
        trends_info = {
            'category': 'Fashion Trends',
            'year': '2026',
            'search_results': results,
            'trend_keywords': [
                'Sustainable Fashion',
                'Minimalism',
                'Vintage & Retro',
                'Bold Colors',
                'Tech Fashion',
                'Luxury Sneakers',
                'Statement Accessories'
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        self.cache['trends'] = trends_info
        return trends_info
    
    def _get_brand_info(self, brand_name: str) -> Dict:
        """Get cached brand information (local database)."""
        brands = {
            'gucci': {
                'name': 'Gucci',
                'founded': 1921,
                'country': 'Italy',
                'headquarters': 'Florence',
                'parent_company': 'Kering',
                'description': 'Luxury Italian fashion house known for GG logos and innovative designs',
                'product_lines': ['Bags', 'Shoes', 'Clothing', 'Watches', 'Jewelry']
            },
            'prada': {
                'name': 'Prada',
                'founded': 1913,
                'country': 'Italy',
                'headquarters': 'Milan',
                'parent_company': 'Independent',
                'description': 'Prestigious Italian luxury fashion brand',
                'product_lines': ['Bags', 'Shoes', 'Clothing', 'Accessories']
            },
            'louis vuitton': {
                'name': 'Louis Vuitton',
                'founded': 1854,
                'country': 'France',
                'headquarters': 'Paris',
                'parent_company': 'LVMH',
                'description': 'World-renowned luxury leather goods and fashion house',
                'product_lines': ['Bags', 'Luggage', 'Shoes', 'Watches', 'Jewelry']
            },
            'chanel': {
                'name': 'Chanel',
                'founded': 1910,
                'country': 'France',
                'headquarters': 'Paris',
                'parent_company': 'Independent',
                'description': 'Iconic French luxury fashion brand',
                'product_lines': ['Bags', 'Clothing', 'Accessories', 'Cosmetics', 'Jewelry']
            },
            'dior': {
                'name': 'Christian Dior',
                'founded': 1946,
                'country': 'France',
                'headquarters': 'Paris',
                'parent_company': 'LVMH',
                'description': 'Luxury fashion house famous for haute couture',
                'product_lines': ['Clothing', 'Bags', 'Shoes', 'Accessories']
            },
            'hermes': {
                'name': 'Hermès',
                'founded': 1837,
                'country': 'France',
                'headquarters': 'Paris',
                'parent_company': 'Independent',
                'description': 'Prestigious luxury goods manufacturer',
                'product_lines': ['Bags', 'Scarves', 'Watches', 'Leather Goods', 'Jewelry']
            },
            'versace': {
                'name': 'Versace',
                'founded': 1978,
                'country': 'Italy',
                'headquarters': 'Milan',
                'parent_company': 'Capri Holdings',
                'description': 'Bold and colorful Italian luxury fashion brand',
                'product_lines': ['Clothing', 'Bags', 'Shoes', 'Accessories']
            }
        }
        
        brand_lower = brand_name.lower()
        if brand_lower in brands:
            return brands[brand_lower]
        else:
            return {
                'name': brand_name,
                'not_found': True,
                'message': f'Information about {brand_name} not in local database. Fetching from internet...'
            }
    
    def _estimate_price(self, product_name: str) -> str:
        """Estimate price range for fashion products."""
        product_lower = product_name.lower()
        
        price_ranges = {
            'handbag': '$1,200 - $25,000+',
            'bag': '$1,000 - $20,000+',
            'shoes': '$400 - $5,000+',
            'dress': '$500 - $10,000+',
            'watch': '$3,000 - $50,000+',
            'jewelry': '$500 - $30,000+',
            'belt': '$300 - $3,000+',
            'scarf': '$200 - $1,000+',
            'sunglasses': '$200 - $800+',
            'wallet': '$300 - $2,000+'
        }
        
        for key, price in price_ranges.items():
            if key in product_lower:
                return price
        
        return '$500 - $5,000+ (Estimated for luxury items)'
    
    def fetch_brand_news(self, brand_name: str) -> List[Dict]:
        """Fetch latest news about a fashion brand."""
        results = self.search_google(f"{brand_name} news fashion 2026")
        return results
    
    def fetch_styling_tips(self, occasion: str) -> List[Dict]:
        """Fetch styling tips for specific occasions."""
        query = f"{occasion} styling tips luxury fashion"
        results = self.search_google(query)
        return results
    
    def get_internet_response(self, user_query: str) -> Dict:
        """
        Get comprehensive internet-based response to user query.
        Analyzes query and fetches relevant data.
        """
        query_lower = user_query.lower()
        
        response = {
            'query': user_query,
            'timestamp': datetime.now().isoformat(),
            'data': []
        }
        
        # Detect query type and fetch appropriate data
        if any(brand in query_lower for brand in ['gucci', 'prada', 'chanel', 'dior', 'louis vuitton', 'hermes']):
            # Brand query
            for brand in ['gucci', 'prada', 'chanel', 'dior', 'louis vuitton', 'hermes']:
                if brand in query_lower:
                    response['data'].append(self.search_fashion_brand(brand))
                    break
        
        elif any(product in query_lower for product in ['handbag', 'bag', 'shoes', 'dress', 'watch', 'jewelry']):
            # Product query
            for product in ['handbag', 'bag', 'shoes', 'dress', 'watch', 'jewelry']:
                if product in query_lower:
                    response['data'].append(self.search_fashion_product(product))
                    break
        
        elif 'trend' in query_lower:
            # Trends
            response['data'].append(self.search_fashion_trends())
        
        elif 'news' in query_lower or 'latest' in query_lower:
            # News
            response['data'].append({
                'category': 'Fashion News',
                'search_results': self.search_google('luxury fashion news 2026')
            })
        
        else:
            # Generic fashion query
            response['data'].append({
                'category': 'Fashion Search',
                'search_results': self.search_google(user_query)
            })
        
        return response

# ==================== Global Fetcher Instance ====================

_fetcher = None

def get_internet_fetcher():
    """Get or create internet fetcher instance."""
    global _fetcher
    if _fetcher is None:
        _fetcher = InternetDataFetcher()
    return _fetcher

def fetch_internet_data(user_query: str) -> Dict:
    """Fetch internet data for a user query."""
    fetcher = get_internet_fetcher()
    return fetcher.get_internet_response(user_query)

# ==================== Testing ====================

if __name__ == "__main__":
    print("=" * 60)
    print("INTERNET DATA FETCHER TEST")
    print("=" * 60 + "\n")
    
    fetcher = InternetDataFetcher()
    
    # Test 1: Brand search
    print("Test 1: Searching for Gucci information...")
    print("-" * 40)
    brand_info = fetcher.search_fashion_brand("Gucci")
    print(f"Brand: {brand_info['brand']}")
    print(f"Local Info: {brand_info['local_info']}")
    print(f"\nSearch Results ({len(brand_info['search_results'])} found):")
    for i, result in enumerate(brand_info['search_results'][:2], 1):
        print(f"  {i}. {result['title']}")
        print(f"     {result['summary'][:100]}...")
    
    # Test 2: Product search
    print("\n" + "=" * 60)
    print("Test 2: Searching for luxury handbags...")
    print("-" * 40)
    product_info = fetcher.search_fashion_product("handbag", "Gucci")
    print(f"Product: {product_info['product']}")
    print(f"Brand: {product_info['brand']}")
    print(f"Estimated Price: {product_info['estimated_price_range']}")
    
    # Test 3: Trends
    print("\n" + "=" * 60)
    print("Test 3: Fetching fashion trends...")
    print("-" * 40)
    trends = fetcher.search_fashion_trends()
    print(f"Category: {trends['category']}")
    print(f"Trend Keywords: {', '.join(trends['trend_keywords'][:5])}")
    
    # Test 4: Generic query
    print("\n" + "=" * 60)
    print("Test 4: General fashion query...")
    print("-" * 40)
    result = fetcher.get_internet_response("Show me luxury fashion trends")
    print(f"Query: {result['query']}")
    print(f"Data Sources: {len(result['data'])}")
    
    print("\n" + "=" * 60)
    print("✓ Internet Data Fetcher Test Complete")
    print("=" * 60)
