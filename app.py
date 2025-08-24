from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import sys
from datetime import datetime
from search_engine import MinimalBlogSearchEngine
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Configuration - Update these paths
MODEL_PATH = "model/gensim-t4d-word2Vec.model"
BLOG_DIR = "blogs/tech4dev_blogs"

# Global search engine instance
search_engine = None

def initialize_search_engine():
    """Initialize the search engine"""
    global search_engine
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model not found: {MODEL_PATH}")
            return False
        
        if not os.path.exists(BLOG_DIR):
            print(f"❌ Blog directory not found: {BLOG_DIR}")
            return False
        
        search_engine = MinimalBlogSearchEngine(MODEL_PATH, BLOG_DIR)
        return True
    except Exception as e:
        print(f"❌ Failed to initialize search engine: {e}")
        return False

# Routes
@app.route('/')
def index():
    """Serve the main search interface"""
    # You can serve the HTML file directly or use render_template
    try:
        with open('blog_search_frontend.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <h1>Blog Search Engine</h1>
        <p>Frontend HTML file not found. Please save the HTML content to 'blog_search_frontend.html'</p>
        <p>Or use the API directly:</p>
        <ul>
            <li>POST /search - {"query": "your search term", "top_k": 5}</li>
            <li>GET /health - Check if the service is running</li>
            <li>GET /stats - Get search engine statistics</li>
        </ul>
        """

@app.route('/search', methods=['POST'])
def search_blogs():
    """Search endpoint for the frontend"""
    if not search_engine:
        return jsonify({
            'error': 'Search engine not initialized',
            'message': 'Please check server logs for initialization errors'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400
        
        query = data['query']
        top_k = data.get('top_k', 5)
        
        if not query.strip():
            return jsonify({'error': 'Empty query'}), 400
        
        print(f"🔍 Searching for: '{query}'")
        
        # Perform search
        results = search_engine.search(query, top_k=top_k)
        
        # Format results for JSON response
        formatted_results = []
        for blog_file, score, metadata in results:
            formatted_results.append([
                blog_file,
                float(score),  # Ensure JSON serializable
                {
                    'title': metadata.get('title', 'Untitled'),
                    'author': metadata.get('author', 'Unknown'),
                    'category': metadata.get('category', 'General'),
                    'url': metadata.get('url', ''),
                    'content_preview': metadata.get('content_preview', 'No preview available')
                }
            ])
        
        return jsonify({
            'query': query,
            'results': formatted_results,
            'total_found': len(formatted_results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return jsonify({
            'error': 'Search failed',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if search_engine else 'unhealthy',
        'search_engine_loaded': search_engine is not None,
        'blog_count': len(search_engine.blog_metadata) if search_engine else 0,
        'vocabulary_size': len(search_engine.model.wv.key_to_index) if search_engine else 0,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get search engine statistics"""
    if not search_engine:
        return jsonify({'error': 'Search engine not initialized'}), 500
    
    return jsonify({
        'blog_count': len(search_engine.blog_metadata),
        'vocabulary_size': len(search_engine.model.wv.key_to_index),
        'indexed_words': len(search_engine.doc_embeddings.items()),
        'sample_vocabulary': list(search_engine.get_vocabulary_sample())[:20],
        'sample_blogs': list(search_engine.blog_metadata.keys())[:10]
    })

@app.route('/suggest', methods=['GET'])
def get_suggestions():
    """Get sample query suggestions"""
    if not search_engine:
        return jsonify([])
    
    # Return some common words from vocabulary as suggestions
    vocab = list(search_engine.model.wv.key_to_index.keys())
    
    # Filter for meaningful words (longer than 4 characters)
    meaningful_words = [word for word in vocab if len(word) > 4 and word.isalpha()]
    
    # Return top 20 as suggestions
    suggestions = meaningful_words[:20]
    
    return jsonify({
        'suggestions': suggestions,
        'total_vocabulary': len(vocab)
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Blog Search Engine Backend...")
    print(f"📍 Model path: {MODEL_PATH}")
    print(f"📁 Blog directory: {BLOG_DIR}")
    
    # Initialize search engine
    if initialize_search_engine():
        print("✅ Search engine initialized successfully!")
        print(f"📊 Loaded {len(search_engine.blog_metadata)} blogs")
        print(f"🔤 Vocabulary size: {len(search_engine.model.wv.key_to_index)}")
        
        print("\n" + "="*50)
        print("🌐 Server starting...")
        print("Frontend: http://localhost:5000")
        print("API: http://localhost:5000/search")
        print("Health: http://localhost:5000/health")
        print("="*50)
        
        # Run the Flask app
        app.run(
            debug=False,      # Set to False in production
            host='0.0.0.0',  # Allow external connections
            port=5000
        )
    else:
        print("❌ Failed to initialize search engine. Please check the paths and try again.")
        print("\n📝 Setup checklist:")
        print(f"  1. Model file exists: {os.path.exists(MODEL_PATH)}")
        print(f"  2. Blog directory exists: {os.path.exists(BLOG_DIR)}")
        print("  3. Run the blog scraper first to download blogs")
        print("  4. Train or load your Word2Vec model")