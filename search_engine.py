import os
import re
from collections import defaultdict
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MinimalBlogSearchEngine:
    def __init__(self,model_path, blog_directory):
        logging.info("Loading word embeddings...")

        self.model=Word2Vec.load(model_path)
        self.blog_directory=blog_directory


        # index: word to list of blog files containing that word
        self.word_to_blogs=defaultdict(list)

        self.blog_metadata={}
        self.doc_embeddings={}
        logging.info("Indexing blog files...")
        self._index_blogs()
        logging.info(f"✅ Indexed {len(self.blog_metadata)} blogs with {len(self.word_to_blogs)} unique words")
        # logging.info(f"Document embeddings: {(self.doc_embeddings)}")

    def _clean_text(self,text):
        return simple_preprocess(text, min_len=3, max_len=50)

    def _create_document_emedding(self, words):
        """Create document embedding by averaging word embeddings"""
        vectors=[]
        for word in words:
            if word in self.model.wv:
                vectors.append(self.model.wv[word])
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.model.wv.vector_size) #return zero vector if no word found

    def _index_blogs(self):
        if not os.path.exists(self.blog_directory):
            logging.error(f"Blog directory not found: {self.blog_directory}")
            return
        
        for filename in os.listdir(self.blog_directory):
            if filename.endswith('.txt'):
                filepath=os.path.join(self.blog_directory,filename)
                self._index_single_blog(filepath)
    
    def _index_single_blog(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content=f.read()
            
            # extract metadata
            lines=content.split('\n')
            metadata={}
            content_start=0

            for i, line in enumerate(lines):
                if line.startswith('Title: '):
                    metadata['title']=line.replace('Title: ','').strip()
                elif line.startswith('Author: '):
                    metadata['author'] = line.replace('Author: ', '').strip()
                elif line.startswith('Category: '):
                    metadata['category'] = line.replace('Category: ', '').strip()
                elif line.startswith('URL: '):
                    metadata['url'] = line.replace('URL: ', '').strip()
                elif line.startswith('==='):
                    content_start = i + 1 #line no. counter to get the content start
                    break
            blog_conent='\n'.join(lines[content_start:])
            metadata['content_preview']=blog_conent[:300]+"..." if len(blog_conent) >300 else blog_conent

            filename=os.path.basename(filepath)
            self.blog_metadata[filename]=metadata

            # tokenize and index words
            all_text=f"{metadata.get('title', '')} {blog_conent}"
            words=self._clean_text(all_text)
            doc_embedding=self._create_document_emedding(words)

            # store the document embeddings
            self.doc_embeddings[filename]=doc_embedding

        except Exception as e:
            logging.error(f"Error indexing {filepath}: {str(e)}")


    def _get_query_embeddings(self, query):
        query_words=self._clean_text(query)
        return self._create_document_emedding(query_words)
    
    def _compute_similarities(self, query_embedding):
        similarities={}

        query_norm=np.linalg.norm(query_embedding)
        if query_norm==0:
            return similarities #for a zero vector, no similar docs found
        
        normalized_query=query_embedding/query_norm
        # compute similarity with each documents
        for filename, doc_embedding in self.doc_embeddings.items():
            doc_norm=np.linalg.norm(doc_embedding)
            if doc_norm>0:
                normalized_doc=doc_embedding/doc_norm
                # cosine similarity
                similarity=np.dot(normalized_query, normalized_doc)
                similarities[filename]=similarity

        return similarities
    def search(self, query, top_k=5):
        """
       Search for blogs using pure embedding based similarity
        
        Args:
            query: Search term
            top_k: Number of results to return
            
        Returns:
            List of tuples: (blog_filename, similarity_score, metadata)
        """
        query=query.lower().strip()
        if not query:
            return self._fallback_search(query,top_k)
        logging.info(f" Searching for: '{query}")

        query_embedding=self._get_query_embeddings(query)

        # compute similarities with all documents
        similarities=self._compute_similarities(query_embedding)
        logging.info(f"The similarities are : {similarities.keys()}")

        if not similarities:
            return []
        
        sorted_results=sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        results=[]
        for filename, score in sorted_results[:top_k]:
            metadata=self.blog_metadata.get(filename, {})
            results.append((filename, score, metadata))
        return results

    def _fallback_search(self, query, top_k):
        """
            Fallback search when the query word in not in vocabulary
        """
        logging.info(f"'{query}' not found in embeddings. Trying text-based search...")

        # simple string matching fallback
        matching_blogs=[]
        query_words=set(self._clean_text(query))

        for blog_file, metadata in self.blog_metadata.items():
            title=metadata.get('title', '').lower()
            content=metadata.get('content_preview', '').lower()
            
            # check if any query words appear in title or content preview
            if any(word in title.lower() or word in content.lower() for word in query_words):
                matching_blogs.append((blog_file, 0.5, metadata))
        
        return matching_blogs[:top_k]
    
    def get_stats(self):
        return {
            'total_blogs':len(self.doc_embeddings),
            'vocabulary_size':len(self.model.wv.key_to_index),
            'embedding_dimension':self.model.wv.vector_size
        }
    def get_vocabulary_sample(self, n=20):
        """Get a sample of words from the vocabulary for testing"""
        vocab = list(self.model.wv.key_to_index.keys())
        return vocab[:n]
    
    def search_interactive(self):
        """Interactive search interface"""
        print("\n" + "="*50)
        print("🔍 MINIMAL BLOG SEARCH ENGINE")
        print("="*50)
        print("Type 'quit' to exit, 'vocab' to see sample vocabulary")
        
        while True:
            query = input("\nSearch: ").strip()
            
            if query.lower() == 'quit':
                break
            elif query.lower() == 'vocab':
                sample_vocab = self.get_vocabulary_sample()
                print(f"Sample vocabulary: {', '.join(sample_vocab)}")
                continue
            elif not query:
                continue
            
            results = self.search(query, top_k=5)
            
            if not results:
                print("No results found.")
                continue
            
            print(f"\n📊 Found {len(results)} results for '{query}':")
            print("-" * 40)
            
            for i, (blog_file, score, metadata) in enumerate(results, 1):
                title = metadata.get('title', 'No title')
                author = metadata.get('author', 'Unknown author')
                category = metadata.get('category', 'No category')
                preview = metadata.get('content_preview', 'No preview')
                url= metadata.get('url', "No URL Found")
                print(f"{i}. {title}")
                print(f"   📁 File: {blog_file}")
                print(f"   👤 Author: {author} | 📂 Category: {category}")
                print(f"   🔗 Similarity: {score:.3f}")
                print(f"   📝 Preview: {preview[:100]}...")
                print(f"   🔗 URL : {url}")
                print()


if __name__=="__main__":
    search_engine=MinimalBlogSearchEngine('model/gensim-t4d-word2Vec.model', 'blogs/tech4dev_blogs')
    stats = search_engine.get_stats()
    print(f"📊 Stats: {stats['total_blogs']} blogs, {stats['vocabulary_size']} vocab, {stats['embedding_dimension']}D vectors")
    
    demo_queries = ["technology", "data", "glific", "artificial intelligence", "development"]
    
    print("\n🔍 Demo searches:")
    for query in demo_queries:
        print(f"\nSearching for: '{query}'")
        results = search_engine.search(query, top_k=3)
        
        if results:
            for i, (blog_file, score, metadata) in enumerate(results,1):
                title = metadata.get('title', 'No title')
                author = metadata.get('author', 'Unknown author')
                category = metadata.get('category', 'No category')
                preview = metadata.get('content_preview', 'No preview')
                url= metadata.get('url', 'No URL Found')
                print(f"{i}. {title}")
                print(f"   📁 File: {blog_file}")
                print(f"   👤 Author: {author} | 📂 Category: {category}")
                print(f"   🔗 Similarity: {score:.3f}")
                print(f"   📝 Preview: {preview[:100]}...")
                print(f"   🔗 URL : {url}")
                print()
        else:
            print("  No results found")
    
    # Interactive mode
    print("\n" + "="*50)
    print("Starting interactive mode...")
    search_engine.search_interactive()