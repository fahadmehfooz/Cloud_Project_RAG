import streamlit as st
import pickle
import faiss
import numpy as np
import os
import time
import logging
import json
import boto3
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RecommenderApp:
    def __init__(self, save_dir='saved_objects'):
        """Initialize the recommender by loading saved components."""
        st.title("🍽️ Restaurant Recommendation System")
        
        # Initialize Bedrock client
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-2",
            aws_access_key_id=st.secrets["aws_access_key_id"],
            aws_secret_access_key=st.secrets["aws_secret_access_key"]
        )
        
        # Load components
        self.load_components(save_dir)
        
    def load_components(self, save_dir):
        """Load saved recommender components."""
        try:
            # Load TF-IDF Vectorizer
            vectorizer_path = os.path.join(save_dir, 'tfidf_vectorizer.pkl')
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load TF-IDF Matrix
            tfidf_matrix_path = os.path.join(save_dir, 'tfidf_matrix.pkl')
            with open(tfidf_matrix_path, 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
            
            # Load FAISS Index
            faiss_index_path = os.path.join(save_dir, 'faiss_index.bin')
            self.faiss_index = faiss.read_index(faiss_index_path)
            
            # Load FAISS Index to Chunk Mapping
            faiss_mapping_path = os.path.join(save_dir, 'faiss_index_to_chunk.pkl')
            with open(faiss_mapping_path, 'rb') as f:
                self.faiss_index_to_chunk = pickle.load(f)
            
            # Load Processed Chunks
            chunks_save_path = os.path.join(save_dir, 'processed_chunks.pkl')
            with open(chunks_save_path, 'rb') as f:
                self.chunks = pickle.load(f)
            
            st.success("Components loaded successfully!")
        except Exception as e:
            st.error(f"Error loading components: {e}")
            logger.error(f"Error loading components: {e}")
    
    def get_titan_embeddings(self, text, dimensions=512):
        """Generate embeddings using Amazon Titan model."""
        logger.info(f"Generating embedding for query")
        
        try:
            body = json.dumps({
                "inputText": text,
                "dimensions": dimensions,
                "normalize": True
            })
            
            response = self.bedrock_client.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                contentType="application/json",
                accept="*/*",
                body=body
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read())
            
            # Extract the embedding
            embedding = response_body['embedding']
            return np.array(embedding, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error generating Titan embeddings: {e}")
            # Generate fallback random embedding if there's an error
            embedding = np.random.rand(dimensions).astype('float32')
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
    
    def retrieve_with_tfidf(self, query, top_n=50):
        """Retrieve documents using TF-IDF similarity."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        try:
            # Transform query
            query_vec = self.vectorizer.transform([query])
            
            # Calculate similarity
            cosine_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Sort and filter results
            scored_chunks = sorted(
                [(i, score) for i, score in enumerate(cosine_sim)],
                key=lambda x: x[1], 
                reverse=True
            )[:top_n]
            
            # Format results
            results = [
                {
                    "metadata": self.chunks[i]["metadata"],
                    "text": self.chunks[i]["text"],
                    "chunk_id": self.chunks[i]["chunk_id"],
                    "score": float(score),
                    "retrieval_method": "tfidf"
                } 
                for i, score in scored_chunks
            ]
            
            return results
        
        except Exception as e:
            st.error(f"Error in TF-IDF retrieval: {e}")
            return []
    
    def retrieve_with_embeddings(self, query_embedding, top_n=20):
        """Retrieve documents using embedding similarity."""
        try:
            # Ensure query embedding is 2D
            query_embedding_2d = query_embedding.reshape(1, -1).astype('float32')
            
            # Search index
            distances, indices = self.faiss_index.search(query_embedding_2d, top_n)
            
            # Format vector results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.faiss_index_to_chunk):
                    chunk = self.faiss_index_to_chunk[idx]
                    if isinstance(chunk.get('text'), dict) and 'text' in chunk['text']:
                        chunk['text'] = chunk['text']['text']
                        
                    results.append({
                        "metadata": chunk,
                        "text": chunk["text"],
                        "chunk_id": chunk["chunk_id"],
                        "score": float(1 / (1 + distances[0][i])),
                        "retrieval_method": "vector"
                    })
            
            return results
            
        except Exception as e:
            st.error(f"Error in vector retrieval: {e}")
            return []
    
    def reciprocal_rank_fusion(self, results_list, k=60.0):
        """Combine multiple result lists using reciprocal rank fusion."""
        # Handle empty results
        if not results_list or all(not results for results in results_list):
            return []
            
        # Deduplicate by unique identifier
        all_results = []
        for results in results_list:
            all_results.extend(results)
            
        # Calculate RRF scores
        item_scores = {}
        
        # Process each result list
        for rank_group_idx, result_list in enumerate(results_list):
            # Get scores by rank position
            for rank, item in enumerate(result_list):
                item_id = item['chunk_id']
                if item_id not in item_scores:
                    item_scores[item_id] = 0.0
                    
                # RRF formula: 1 / (k + rank)
                item_scores[item_id] += 1.0 / (k + rank)
        
        # Apply RRF scores to results
        for item in all_results:
            item['rrf_score'] = item_scores.get(item['chunk_id'], 0.0)
            item['original_score'] = item['score']  # Preserve original score
            item['score'] = item['rrf_score']  # Replace with fusion score
            
        # Sort by fusion score
        return sorted(all_results, key=lambda x: x['rrf_score'], reverse=True)
    
    def call_bedrock_claude(self, prompt):
        """Call AWS Bedrock Claude model to generate a response."""
        try:
            payload = {
                "modelId": st.secrets["inference_profile"],
                "body": json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 500,
                    "temperature": 0.3,
                    "system": """You are a helpful restaurant recommendation assistant. 
                    Provide concise, helpful recommendations based on the retrieved information.
                    Focus on menu items, price, location, and cuisine types that best match the query.""",
                    "messages": [{
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": prompt
                        }]
                    }]
                })
            }
            
            # Call the model using invoke_model
            response = self.bedrock_client.invoke_model(**payload)
            
            # Parse the response
            result = json.loads(response["body"].read().decode("utf-8"))
            generated_text = result.get("content", [{}])[0].get("text", "").strip()
            
            return generated_text
            
        except Exception as e:
            st.error(f"Error calling AWS Bedrock Claude: {e}")
            return None
    
    def format_context(self, results):
        """Format retrieval results into a context string for the LLM."""
        context = "Here are some relevant restaurant items:\n\n"
        
        for i, res in enumerate(results):
            meta = res.get('metadata', {})
            context += f"[{i+1}] {meta.get('restaurant_name', 'Unknown')}: {meta.get('menu_item', 'Unknown Item')}\n"
            context += f"Description: {meta.get('menu_description', 'No description available')}\n"
            context += f"Price: {meta.get('price_tier', 'unknown').capitalize()}\n"
            
            if meta.get('cuisine_types'):
                context += f"Cuisine: {', '.join(meta.get('cuisine_types', []))}\n"
                
            if meta.get('ingredients'):
                context += f"Ingredients: {', '.join(meta.get('ingredients', []))}\n"
                
            context += f"Location: {meta.get('location', 'Unknown')}\n\n"
        
        return context
    
    def search(self, query, top_k=5):
        """Perform hybrid search with the given query."""
        # Step 1: Retrieve with TF-IDF
        tfidf_results = self.retrieve_with_tfidf(query, 50)
        
        # Step 2: Generate query embedding
        try:
            query_embedding = self.get_titan_embeddings(query)
            
            # Step 3: Search with vector embedding
            vector_results = self.retrieve_with_embeddings(query_embedding, 20)
            
            # Step 4: Combine results with RRF
            results_list = [tfidf_results]
            if vector_results:
                results_list.append(vector_results)
                
            combined = self.reciprocal_rank_fusion(results_list)
            
        except Exception as e:
            st.error(f"Error in vector search: {e}")
            # Fallback to TF-IDF results only
            combined = tfidf_results
        
        # Sort by score and limit results
        return sorted(combined, key=lambda x: x.get("score", 0), reverse=True)[:top_k]
    
    def generate_recommendation(self, query, results):
        """Generate a detailed recommendation response."""
        # Format the context and prompt for the LLM
        context = self.format_context(results)
        prompt = (
            f"Based on the following restaurant information, provide a helpful response to this query: '{query}'\n\n"
            f"{context}\n"
            f"Provide a concise recommendation that highlights the best matches for the query. "
            f"Include details about the food, price, and location."
        )
        
        # Try calling the LLM
        llm_response = self.call_bedrock_claude(prompt)
        
        # If the LLM call fails, generate a simple response
        if llm_response is None:
            llm_response = self.simple_fallback_response(query, results)
        
        return llm_response
    
    def simple_fallback_response(self, query, results):
        """Generate a simple response if LLM call fails."""
        if not results:
            return f"I couldn't find any restaurant recommendations for '{query}'. Could you provide more details or try a different search?"
        
        response = f"Here are some restaurant recommendations for '{query}':\n\n"
        
        for i, result in enumerate(results[:3]):
            meta = result.get('metadata', {})
            response += (
                f"{i+1}. {meta.get('restaurant_name', 'Unknown')}\n"
                f"   - Menu Item: {meta.get('menu_item', 'N/A')}\n"
                f"   - Price: {meta.get('price_tier', '').capitalize()}\n"
            )
            
            if meta.get('cuisine_types'):
                response += f"   - Cuisine: {', '.join(meta.get('cuisine_types', []))}\n"
                
            if meta.get('ingredients'):
                response += f"   - Ingredients: {', '.join(meta.get('ingredients', []))}\n"
                
            response += f"   - Location: {meta.get('location', 'Unknown')}\n\n"
        
        return response
    
    def display_results(self, results):
        """Display search results in a formatted manner."""
        for i, result in enumerate(results, 1):
            meta = result.get('metadata', {})
            
            # Create an expandable section for each result
            with st.expander(f"{i}. {meta.get('restaurant_name', 'Unknown Restaurant')} - {meta.get('menu_item', 'Unknown Item')}"):
                st.write(f"**Restaurant:** {meta.get('restaurant_name', 'N/A')}")
                st.write(f"**Menu Item:** {meta.get('menu_item', 'N/A')}")
                st.write(f"**Description:** {meta.get('menu_description', 'No description available')}")
                
                # Price Tier
                price_tier = meta.get('price_tier', '').capitalize()
                st.write(f"**Price Tier:** {price_tier}")
                
                # Cuisine Types
                cuisines = meta.get('cuisine_types', [])
                if cuisines:
                    st.write(f"**Cuisine:** {', '.join(cuisines)}")
                
                # Ingredients
                ingredients = meta.get('ingredients', [])
                if ingredients:
                    st.write(f"**Ingredients:** {', '.join(ingredients)}")
                
                # Location
                st.write(f"**Location:** {meta.get('location', 'Unknown')}")
                
                # Retrieval Method and Score
                st.write(f"**Retrieval Method:** {result.get('retrieval_method', 'N/A')}")
                st.write(f"**Relevance Score:** {result.get('score', 0):.4f}")

def main():

     
    # Initialize the app
    app = RecommenderApp()
    
    # Search interface
    st.header("Search for Restaurant Recommendations")
    
    # Query input
    query = st.text_input("Enter your restaurant or food preference:", 
                           placeholder="e.g., spicy vegetarian food in san francisco")
    
    # Number of results slider
    top_k = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)
    
    # Search button
    if st.button("Search Recommendations"):
        if query:
            # Performance tracking
            start_time = time.time()
            
            try:
                # Perform search to get results
                results = app.search(query, top_k)
                
                # Generate LLM recommendation
                if results:
                    # Display LLM-generated recommendation
                    st.subheader("AI Recommendation")
                    llm_response = app.generate_recommendation(query, results)
                    st.write(llm_response)
                    
                    # Divider
                    st.markdown("---")
                    
                    # Display detailed results
                    st.subheader("Detailed Restaurant Matches")
                    app.display_results(results)
                else:
                    st.warning("No results found. Try a different search query.")
                
                # Calculate execution time
                execution_time = time.time() - start_time
                st.info(f"Search completed in {execution_time:.2f} seconds")
                
            except Exception as e:
                st.error(f"An error occurred during search: {e}")
        else:
            st.warning("Please enter a search query.")

if __name__ == "__main__":
    main()
