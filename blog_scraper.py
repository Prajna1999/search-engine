import requests
from bs4 import BeautifulSoup
import os
import time
import re
from urllib.parse import urljoin, urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BlogScraper:
    def __init__(self, base_url="https://projecttech4dev.org/blogs/", max_pages=45):
        self.base_url = base_url
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Create directory for blog files
        self.output_dir = "tech4dev_blogs"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created directory: {self.output_dir}")

    def clean_filename(self, title):
        """Clean title to make it a valid filename"""
        # Remove HTML tags if any
        title = re.sub(r'<[^>]+>', '', title)
        # Replace invalid characters
        title = re.sub(r'[<>:"/\\|?*]', '_', title)
        # Remove extra whitespace and limit length
        title = ' '.join(title.split())
        title = title[:100]  # Limit to 100 characters
        return title.strip()

    def get_page_content(self, url):
        """Get page content with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch {url} after {max_retries} attempts")
                    return None

    def extract_blog_links_from_page(self, soup, base_url):
        """Extract blog post links from the current page using the actual HTML structure"""
        blog_links = []
        
        # Look for the loop container that contains all blog items
        loop_container = soup.find('div', class_='elementor-loop-container')
        if not loop_container:
            logger.warning("Could not find elementor-loop-container")
            return blog_links
        
        # Find all blog items within the loop container
        blog_items = loop_container.find_all('div', class_='e-loop-item')
        
        for item in blog_items:
            # Look for the clickable container with data-ha-element-link attribute
            clickable_div = item.find('div', {'data-ha-element-link': True})
            if clickable_div:
                # Extract URL from the data attribute
                link_data = clickable_div.get('data-ha-element-link')
                if link_data:
                    # Parse the JSON-like data to extract URL
                    try:
                        import json
                        link_info = json.loads(link_data.replace("\\", ""))
                        blog_url = link_info.get('url')
                        
                        if blog_url:
                            # Find the title within the heading element
                            title_element = item.find('h3', class_='elementor-heading-title')
                            if title_element:
                                title = title_element.get_text(strip=True)
                                
                                # Find the category for additional context
                                category_element = item.find('span', class_='elementor-post-info__terms-list-item')
                                category = category_element.get_text(strip=True) if category_element else "Unknown"
                                
                                # Find the author and date
                                author_element = item.find('span', class_='elementor-post-info__item--type-author')
                                author = author_element.get_text(strip=True) if author_element else "Unknown"
                                
                                date_element = item.find('span', class_='elementor-post-info__item--type-date')
                                date = date_element.get_text(strip=True) if date_element else "Unknown"
                                
                                blog_links.append({
                                    'url': blog_url,
                                    'title': title,
                                    'category': category,
                                    'author': author,
                                    'date': date
                                })
                                
                                logger.debug(f"Found blog: {title} by {author} in {category}")
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Could not parse link data: {e}")
                        continue
        
        logger.info(f"Extracted {len(blog_links)} blog links from current page")
        return blog_links

    def extract_blog_content(self, url, title):
        """Extract main content from a blog post"""
        response = self.get_page_content(url)
        if not response:
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
            element.decompose()
        
        # Try to find the main content using various selectors specific to this site
        content_selectors = [
            'article .elementor-widget-theme-post-content',
            'article .entry-content',
            '.elementor-widget-theme-post-content .elementor-widget-container',
            'article .post-content',
            '.single-post-content',
            'main .post-content',
            '[data-elementor-type="single"] .elementor-widget-theme-post-content',
            '.elementor-widget-theme-post-content',
            'article',
            'main'
        ]
        
        content = None
        for selector in content_selectors:
            elements = soup.select(selector)
            for element in elements:
                text_content = element.get_text(separator='\n', strip=True)
                if len(text_content) > 500:  # Ensure we have substantial content
                    content = text_content
                    logger.debug(f"Found content using selector: {selector}")
                    break
            if content:
                break
        
        # Fallback: try to get content from the entire page body
        if not content or len(content) < 500:
            # Look for post title and content around it
            body = soup.find('body')
            if body:
                # Remove common non-content elements
                for remove_class in ['elementor-nav-menu', 'elementor-button', 'pagination', 'sidebar', 'widget']:
                    for elem in body.find_all(class_=re.compile(remove_class)):
                        elem.decompose()
                
                content = body.get_text(separator='\n', strip=True)
                
                # Clean up the content to remove excessive whitespace
                lines = content.split('\n')
                cleaned_lines = []
                for line in lines:
                    line = line.strip()
                    if line and len(line) > 1:  # Skip empty lines and single characters
                        cleaned_lines.append(line)
                
                content = '\n'.join(cleaned_lines)
        
        return content

    def save_blog_content(self, blog_data, content):
        """Save blog content to a text file"""
        if not content or len(content.strip()) < 100:
            logger.warning(f"Content too short for: {blog_data['title']}")
            return False
            
        filename = self.clean_filename(blog_data['title']) + ".txt"
        filepath = os.path.join(self.output_dir, filename)
        
        # Handle duplicate filenames
        counter = 1
        while os.path.exists(filepath):
            name, ext = os.path.splitext(filename)
            filepath = os.path.join(self.output_dir, f"{name}_{counter}{ext}")
            counter += 1
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Title: {blog_data['title']}\n")
                f.write(f"Author: {blog_data['author']}\n")
                f.write(f"Date: {blog_data['date']}\n")
                f.write(f"Category: {blog_data['category']}\n")
                f.write(f"URL: {blog_data['url']}\n")
                f.write(f"{'='*50}\n\n")
                f.write(content)
            
            logger.info(f"Saved: {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save {filename}: {str(e)}")
            return False

    def scrape_blogs(self):
        """Main method to scrape all blog posts"""
        logger.info(f"Starting to scrape blogs from {self.base_url}")
        logger.info(f"Will process up to {self.max_pages} pages")
        
        total_downloaded = 0
        all_blog_links = []
        
        # First, collect all blog links from all pages
        for page_num in range(1, self.max_pages + 1):
            if page_num == 1:
                page_url = self.base_url
            else:
                # Based on the HTML structure, pagination uses /page/N/ format
                page_url = f"{self.base_url}page/{page_num}/"
            
            logger.info(f"Fetching page {page_num}: {page_url}")
            
            response = self.get_page_content(page_url)
            if not response:
                logger.error(f"Failed to fetch page {page_num}")
                continue
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract blog links from this page
            blog_links = self.extract_blog_links_from_page(soup, page_url)
            
            if not blog_links:
                logger.info(f"No blog links found on page {page_num}")
                # Check if we've reached the end by looking for pagination
                pagination = soup.find('nav', class_='elementor-pagination')
                if pagination:
                    current_page = pagination.find('span', class_='current')
                    if current_page and str(page_num) in current_page.get_text():
                        # We're on a valid page but no content found
                        logger.info(f"Reached end of content at page {page_num}")
                        break
                else:
                    logger.info(f"No pagination found, assuming end of content at page {page_num}")
                    break
            
            all_blog_links.extend(blog_links)
            logger.info(f"Found {len(blog_links)} blog links on page {page_num} (Total so far: {len(all_blog_links)})")
            
            # Add delay between page requests
            time.sleep(2)
        
        # Remove duplicates from all collected links
        seen_urls = set()
        unique_blog_links = []
        for link in all_blog_links:
            if link['url'] not in seen_urls:
                seen_urls.add(link['url'])
                unique_blog_links.append(link)
        
        logger.info(f"Total unique blog posts found: {len(unique_blog_links)}")
        
        # Now download content for each blog post
        for i, blog_data in enumerate(unique_blog_links, 1):
            url = blog_data['url']
            title = blog_data['title']
            
            logger.info(f"Processing blog {i}/{len(unique_blog_links)}: {title}")
            
            content = self.extract_blog_content(url, title)
            if content:
                if self.save_blog_content(blog_data, content):
                    total_downloaded += 1
                else:
                    logger.warning(f"Failed to save: {title}")
            else:
                logger.warning(f"Could not extract content from: {url}")
            
            # Add delay between requests to be respectful
            time.sleep(3)
        
        logger.info(f"Scraping completed! Downloaded {total_downloaded} blog posts to '{self.output_dir}' directory")
        return total_downloaded

def main():
    """Main function to run the scraper"""
    scraper = BlogScraper(
        base_url="https://projecttech4dev.org/blogs/",
        max_pages=45
    )
    
    try:
        total_downloaded = scraper.scrape_blogs()
        print(f"\nScraping completed successfully!")
        print(f"Total blogs downloaded: {total_downloaded}")
        print(f"Files saved in: {scraper.output_dir}")
        
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()