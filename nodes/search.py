import concurrent.futures
from typing import List
from tavily import TavilyClient
from schemas.document_schema import Document
from utils.config import TAVILY_API_KEY
import uuid
import trafilatura
from urllib.parse import urlparse

def search_single_query(query: str, client: TavilyClient) -> List[Document]:
    try:
        results = client.search(query, search_depth="advanced", include_raw_content=True)
        documents = []
        for res in results.get('results', []):
            documents.append(Document(
                id=str(uuid.uuid4()),
                url=res.get('url'),
                title=res.get('title'),
                raw_html=res.get('raw_content')
            ))
        return documents
    except Exception as e:
        print(f"Error searching for query '{query}': {e}")
        return []

def search(sub_queries: List[str]) -> List[Document]:
    client = TavilyClient(api_key=TAVILY_API_KEY)
    all_documents = []
    seen_urls = set()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_query = {executor.submit(search_single_query, q, client): q for q in sub_queries}
        for future in concurrent.futures.as_completed(future_to_query):
            docs = future.result()
            for doc in docs:
                if doc.url not in seen_urls:
                    all_documents.append(doc)
                    seen_urls.add(doc.url)
                    
    return all_documents


def fetch_manual_documents(urls: List[str]) -> List[Document]:
    documents: List[Document] = []
    seen = set()

    for raw_url in urls:
        url = (raw_url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)

        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                continue

            domain = urlparse(url).netloc or "Manual Source"
            documents.append(
                Document(
                    id=str(uuid.uuid4()),
                    url=url,
                    title=domain,
                    raw_html=downloaded,
                )
            )
        except Exception as e:
            print(f"Error fetching manual URL '{url}': {e}")

    return documents
