# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tools for the MemoryContextAgent."""

import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from google.adk.tools import ToolContext
from utils.supabase_client import supabase_manager
import logging

logger = logging.getLogger(__name__)


async def mcp_memory_store_tool(
    action: str,
    memory_key: str = "",
    memory_content: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Persistent conversation and context storage via MCP servers.
    
    Args:
        action: Action to perform (store, retrieve, update, delete, search)
        memory_key: Unique key for memory item
        memory_content: Content to store
        metadata: Additional metadata
        tool_context: ADK tool context
    
    Returns:
        Dict containing memory operation result
    """
    try:
        if action == "store":
            # Store memory content
            if not memory_content:
                return {"error": "Memory content required for storage"}
            
            # Generate content hash for deduplication
            content_hash = generate_content_hash(memory_content)
            
            # Check for existing content
            existing = await supabase_manager.query_records(
                "memory_store",
                filters={"content_hash": content_hash}
            )
            
            if existing and not metadata.get("force_duplicate", False):
                return {
                    "action": "store",
                    "memory_key": memory_key,
                    "status": "duplicate",
                    "existing_id": existing[0]["id"]
                }
            
            # Create memory record
            memory_record = {
                "id": str(uuid.uuid4()),
                "memory_key": memory_key,
                "content": memory_content,
                "content_hash": content_hash,
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat(),
                "access_count": 0,
                "last_accessed": datetime.now().isoformat()
            }
            
            await supabase_manager.insert_record("memory_store", memory_record)
            
            # Generate embedding for semantic search
            await generate_and_store_embedding(memory_record["id"], memory_content)
            
            return {
                "action": "store",
                "memory_key": memory_key,
                "memory_id": memory_record["id"],
                "content_hash": content_hash,
                "success": True
            }
        
        elif action == "retrieve":
            # Retrieve memory by key
            memories = await supabase_manager.query_records(
                "memory_store",
                filters={"memory_key": memory_key}
            )
            
            if not memories:
                return {
                    "action": "retrieve",
                    "memory_key": memory_key,
                    "found": False,
                    "content": None
                }
            
            memory = memories[0]
            
            # Update access statistics
            await supabase_manager.update_record(
                "memory_store",
                memory["id"],
                {
                    "access_count": memory.get("access_count", 0) + 1,
                    "last_accessed": datetime.now().isoformat()
                }
            )
            
            return {
                "action": "retrieve",
                "memory_key": memory_key,
                "found": True,
                "content": memory["content"],
                "metadata": memory.get("metadata", {}),
                "created_at": memory["created_at"]
            }
        
        elif action == "search":
            # Search memories by content
            search_query = metadata.get("query", "") if metadata else ""
            limit = metadata.get("limit", 10) if metadata else 10
            
            if not search_query:
                return {"error": "Search query required"}
            
            # Perform semantic search
            search_results = await semantic_memory_search(search_query, limit)
            
            return {
                "action": "search",
                "query": search_query,
                "results": search_results,
                "count": len(search_results)
            }
        
        elif action == "update":
            # Update existing memory
            if not memory_content:
                return {"error": "Memory content required for update"}
            
            memories = await supabase_manager.query_records(
                "memory_store",
                filters={"memory_key": memory_key}
            )
            
            if not memories:
                return {"error": "Memory not found for update"}
            
            memory_id = memories[0]["id"]
            new_content_hash = generate_content_hash(memory_content)
            
            await supabase_manager.update_record(
                "memory_store",
                memory_id,
                {
                    "content": memory_content,
                    "content_hash": new_content_hash,
                    "metadata": metadata or memories[0].get("metadata", {}),
                    "updated_at": datetime.now().isoformat()
                }
            )
            
            # Update embedding
            await generate_and_store_embedding(memory_id, memory_content)
            
            return {
                "action": "update",
                "memory_key": memory_key,
                "memory_id": memory_id,
                "success": True
            }
        
        elif action == "delete":
            # Delete memory
            memories = await supabase_manager.query_records(
                "memory_store",
                filters={"memory_key": memory_key}
            )
            
            if not memories:
                return {"error": "Memory not found for deletion"}
            
            memory_id = memories[0]["id"]
            
            # Delete from memory store
            await supabase_manager.delete_record("memory_store", memory_id)
            
            # Delete associated embeddings
            await supabase_manager.delete_record("memory_embeddings", memory_id)
            
            return {
                "action": "delete",
                "memory_key": memory_key,
                "memory_id": memory_id,
                "success": True
            }
        
        else:
            return {"error": f"Unknown action: {action}"}
            
    except Exception as e:
        logger.error(f"Error in MCP memory store tool: {e}")
        return {"error": str(e), "action": action}


def generate_content_hash(content: Dict[str, Any]) -> str:
    """Generate hash for content deduplication."""
    content_str = json.dumps(content, sort_keys=True)
    return hashlib.md5(content_str.encode()).hexdigest()


async def generate_and_store_embedding(memory_id: str, content: Dict[str, Any]):
    """Generate and store embedding for semantic search."""
    
    # Extract text content for embedding
    text_content = extract_text_content(content)
    
    if text_content:
        # Generate embedding (simulate with hash for now)
        # In production, use actual embedding model like text-embedding-ada-002
        embedding = generate_mock_embedding(text_content)
        
        # Store embedding
        embedding_record = {
            "id": memory_id,
            "memory_id": memory_id,
            "embedding": embedding,
            "text_content": text_content[:1000],  # Store first 1000 chars
            "created_at": datetime.now().isoformat()
        }
        
        await supabase_manager.insert_record("memory_embeddings", embedding_record)


def extract_text_content(content: Dict[str, Any]) -> str:
    """Extract text content from memory for embedding generation."""
    
    text_parts = []
    
    # Extract from common fields
    for field in ["text", "content", "message", "description", "summary"]:
        if field in content and isinstance(content[field], str):
            text_parts.append(content[field])
    
    # Extract from nested structures
    for key, value in content.items():
        if isinstance(value, str) and len(value) > 10:
            text_parts.append(f"{key}: {value}")
        elif isinstance(value, dict):
            nested_text = extract_text_content(value)
            if nested_text:
                text_parts.append(nested_text)
    
    return " ".join(text_parts)


def generate_mock_embedding(text: str) -> List[float]:
    """Generate mock embedding for demonstration (replace with actual model)."""
    
    # Simple hash-based mock embedding
    hash_value = hashlib.md5(text.encode()).hexdigest()
    
    # Convert hex to float values between -1 and 1
    embedding = []
    for i in range(0, len(hash_value), 2):
        hex_pair = hash_value[i:i+2]
        value = (int(hex_pair, 16) / 255.0) * 2 - 1  # Normalize to [-1, 1]
        embedding.append(round(value, 6))
    
    # Pad or truncate to 1536 dimensions (OpenAI embedding size)
    while len(embedding) < 1536:
        embedding.extend(embedding[:min(16, 1536 - len(embedding))])
    
    return embedding[:1536]


async def semantic_memory_search(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Perform semantic search across memory embeddings."""
    
    # Generate query embedding
    query_embedding = generate_mock_embedding(query)
    
    # In production, use pgvector for similarity search
    # For now, simulate with text search
    memories = await supabase_manager.query_records("memory_store", limit=limit * 2)
    
    # Simple text-based relevance scoring
    scored_memories = []
    query_words = set(query.lower().split())
    
    for memory in memories:
        content_text = extract_text_content(memory.get("content", {})).lower()
        content_words = set(content_text.split())
        
        # Calculate word overlap score
        overlap = len(query_words.intersection(content_words))
        relevance_score = overlap / max(len(query_words), 1)
        
        if relevance_score > 0:
            scored_memories.append({
                "memory": memory,
                "relevance_score": relevance_score
            })
    
    # Sort by relevance and return top results
    scored_memories.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return [{
        "memory_id": item["memory"]["id"],
        "memory_key": item["memory"]["memory_key"],
        "content": item["memory"]["content"],
        "relevance_score": item["relevance_score"],
        "created_at": item["memory"]["created_at"]
    } for item in scored_memories[:limit]]


async def supabase_vector_tool(
    action: str,
    collection_name: str = "default",
    vector_data: Dict[str, Any] = None,
    query_vector: List[float] = None,
    similarity_threshold: float = 0.7,
    limit: int = 10,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Vector embeddings and semantic search using Supabase pgvector extension.
    
    Args:
        action: Action to perform (store, search, update, delete)
        collection_name: Collection name for organizing vectors
        vector_data: Data to store with vector
        query_vector: Vector to search with
        similarity_threshold: Minimum similarity for results
        limit: Maximum number of results
        tool_context: ADK tool context
    
    Returns:
        Dict containing vector operation result
    """
    try:
        if action == "store":
            # Store vector with data
            if not vector_data:
                return {"error": "Vector data required for storage"}
            
            # Generate embedding for the content
            content_text = extract_text_content(vector_data)
            embedding = generate_mock_embedding(content_text) if content_text else []
            
            vector_record = {
                "id": str(uuid.uuid4()),
                "collection_name": collection_name,
                "data": vector_data,
                "embedding": embedding,
                "content_text": content_text[:2000],  # Store first 2000 chars
                "created_at": datetime.now().isoformat()
            }
            
            await supabase_manager.insert_record("vector_store", vector_record)
            
            return {
                "action": "store",
                "collection_name": collection_name,
                "vector_id": vector_record["id"],
                "embedding_size": len(embedding),
                "success": True
            }
        
        elif action == "search":
            # Search for similar vectors
            if not query_vector:
                # Generate embedding from search text
                search_text = vector_data.get("search_text", "") if vector_data else ""
                if not search_text:
                    return {"error": "Query vector or search text required"}
                query_vector = generate_mock_embedding(search_text)
            
            # Get all vectors in collection for similarity calculation
            vectors = await supabase_manager.query_records(
                "vector_store",
                filters={"collection_name": collection_name}
            )
            
            # Calculate similarities
            similar_vectors = []
            for vector_record in vectors:
                stored_embedding = vector_record.get("embedding", [])
                if stored_embedding:
                    similarity = calculate_cosine_similarity(query_vector, stored_embedding)
                    
                    if similarity >= similarity_threshold:
                        similar_vectors.append({
                            "vector_id": vector_record["id"],
                            "data": vector_record["data"],
                            "similarity": similarity,
                            "created_at": vector_record["created_at"]
                        })
            
            # Sort by similarity and limit results
            similar_vectors.sort(key=lambda x: x["similarity"], reverse=True)
            
            return {
                "action": "search",
                "collection_name": collection_name,
                "query_embedding_size": len(query_vector),
                "results": similar_vectors[:limit],
                "total_found": len(similar_vectors)
            }
        
        elif action == "update":
            # Update vector data
            vector_id = vector_data.get("vector_id") if vector_data else ""
            if not vector_id:
                return {"error": "Vector ID required for update"}
            
            # Generate new embedding
            content_text = extract_text_content(vector_data)
            embedding = generate_mock_embedding(content_text) if content_text else []
            
            await supabase_manager.update_record(
                "vector_store",
                vector_id,
                {
                    "data": vector_data,
                    "embedding": embedding,
                    "content_text": content_text[:2000],
                    "updated_at": datetime.now().isoformat()
                }
            )
            
            return {
                "action": "update",
                "vector_id": vector_id,
                "embedding_size": len(embedding),
                "success": True
            }
        
        elif action == "delete":
            # Delete vector
            vector_id = vector_data.get("vector_id") if vector_data else ""
            if not vector_id:
                return {"error": "Vector ID required for deletion"}
            
            await supabase_manager.delete_record("vector_store", vector_id)
            
            return {
                "action": "delete",
                "vector_id": vector_id,
                "success": True
            }
        
        else:
            return {"error": f"Unknown action: {action}"}
            
    except Exception as e:
        logger.error(f"Error in vector tool: {e}")
        return {"error": str(e), "action": action}


def calculate_cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    
    if len(vector1) != len(vector2):
        return 0.0
    
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    
    # Calculate magnitudes
    magnitude1 = sum(a * a for a in vector1) ** 0.5
    magnitude2 = sum(b * b for b in vector2) ** 0.5
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # Calculate cosine similarity
    return dot_product / (magnitude1 * magnitude2)


async def realtime_context_tool(
    context_id: str,
    context_update: Dict[str, Any],
    update_type: str = "append",
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Live context synthesis and updates with real-time notifications.
    
    Args:
        context_id: Context identifier
        context_update: Context update data
        update_type: Type of update (append, replace, merge)
        tool_context: ADK tool context
    
    Returns:
        Dict containing context update result
    """
    try:
        timestamp = datetime.now().isoformat()
        
        # Get current context
        current_context = await supabase_manager.get_record("context_store", context_id)
        
        if current_context:
            # Update existing context
            existing_data = current_context.get("data", {})
            
            if update_type == "append":
                # Append to existing context
                if isinstance(existing_data, list):
                    updated_data = existing_data + [context_update]
                else:
                    updated_data = [existing_data, context_update]
            elif update_type == "merge":
                # Merge with existing context
                if isinstance(existing_data, dict) and isinstance(context_update, dict):
                    updated_data = {**existing_data, **context_update}
                else:
                    updated_data = context_update
            else:  # replace
                updated_data = context_update
            
            await supabase_manager.update_record(
                "context_store",
                context_id,
                {
                    "data": updated_data,
                    "updated_at": timestamp,
                    "update_count": current_context.get("update_count", 0) + 1
                }
            )
        else:
            # Create new context
            context_record = {
                "id": context_id,
                "data": context_update,
                "created_at": timestamp,
                "updated_at": timestamp,
                "update_count": 1
            }
            
            await supabase_manager.insert_record("context_store", context_record)
            updated_data = context_update
        
        # Broadcast context update
        await supabase_manager.broadcast_message(
            "context_updates",
            {
                "type": "context_updated",
                "context_id": context_id,
                "update_type": update_type,
                "timestamp": timestamp,
                "preview": str(updated_data)[:200]  # First 200 chars
            }
        )
        
        return {
            "context_id": context_id,
            "update_type": update_type,
            "timestamp": timestamp,
            "data_size": len(str(updated_data)),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error in realtime context tool: {e}")
        return {"error": str(e), "context_id": context_id}


async def context_synthesis_tool(
    synthesis_request: Dict[str, Any],
    synthesis_mode: str = "comprehensive",
    max_context_items: int = 50,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Intelligent context aggregation and ranking from multiple sources.
    
    Args:
        synthesis_request: Request with sources and criteria
        synthesis_mode: Mode (comprehensive, focused, summarized)
        max_context_items: Maximum context items to include
        tool_context: ADK tool context
    
    Returns:
        Dict containing synthesized context
    """
    try:
        synthesis_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Extract synthesis parameters
        query = synthesis_request.get("query", "")
        sources = synthesis_request.get("sources", ["memory_store", "context_store"])
        filters = synthesis_request.get("filters", {})
        ranking_criteria = synthesis_request.get("ranking_criteria", ["relevance", "recency"])
        
        # Collect context from sources
        collected_contexts = []
        
        for source in sources:
            if source == "memory_store":
                # Search memory store
                if query:
                    search_results = await semantic_memory_search(query, max_context_items)
                    for result in search_results:
                        collected_contexts.append({
                            "source": "memory_store",
                            "content": result["content"],
                            "relevance_score": result["relevance_score"],
                            "created_at": result["created_at"],
                            "memory_id": result["memory_id"]
                        })
            
            elif source == "context_store":
                # Search context store
                contexts = await supabase_manager.query_records("context_store", limit=max_context_items)
                for context in contexts:
                    content_text = extract_text_content(context.get("data", {}))
                    relevance_score = calculate_text_relevance(query, content_text) if query else 0.5
                    
                    collected_contexts.append({
                        "source": "context_store",
                        "content": context["data"],
                        "relevance_score": relevance_score,
                        "created_at": context["created_at"],
                        "context_id": context["id"]
                    })
        
        # Apply filters
        filtered_contexts = apply_context_filters(collected_contexts, filters)
        
        # Rank and select top contexts
        ranked_contexts = rank_contexts(filtered_contexts, ranking_criteria)
        selected_contexts = ranked_contexts[:max_context_items]
        
        # Synthesize final context
        synthesized_context = synthesize_context_items(selected_contexts, synthesis_mode)
        
        # Calculate synthesis metrics
        synthesis_metrics = {
            "total_sources_searched": len(sources),
            "contexts_collected": len(collected_contexts),
            "contexts_after_filtering": len(filtered_contexts),
            "contexts_selected": len(selected_contexts),
            "synthesis_mode": synthesis_mode,
            "average_relevance": sum(ctx["relevance_score"] for ctx in selected_contexts) / max(len(selected_contexts), 1)
        }
        
        result = {
            "synthesis_id": synthesis_id,
            "timestamp": timestamp,
            "query": query,
            "synthesized_context": synthesized_context,
            "context_sources": selected_contexts,
            "synthesis_metrics": synthesis_metrics
        }
        
        # Store synthesis result
        await supabase_manager.insert_record(
            "context_synthesis",
            {
                "id": synthesis_id,
                "synthesis_result": result,
                "timestamp": timestamp
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in context synthesis: {e}")
        return {"error": str(e), "synthesis_id": synthesis_id}


def calculate_text_relevance(query: str, text: str) -> float:
    """Calculate relevance score between query and text."""
    
    if not query or not text:
        return 0.0
    
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    
    # Calculate word overlap
    overlap = len(query_words.intersection(text_words))
    return overlap / max(len(query_words), 1)


def apply_context_filters(contexts: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Apply filters to context items."""
    
    filtered = contexts
    
    # Date range filter
    if "date_range" in filters:
        start_date = filters["date_range"].get("start")
        end_date = filters["date_range"].get("end")
        
        if start_date or end_date:
            date_filtered = []
            for ctx in filtered:
                ctx_date = datetime.fromisoformat(ctx["created_at"].replace('Z', '+00:00'))
                
                if start_date and ctx_date < datetime.fromisoformat(start_date):
                    continue
                if end_date and ctx_date > datetime.fromisoformat(end_date):
                    continue
                
                date_filtered.append(ctx)
            filtered = date_filtered
    
    # Relevance threshold filter
    if "min_relevance" in filters:
        min_relevance = filters["min_relevance"]
        filtered = [ctx for ctx in filtered if ctx["relevance_score"] >= min_relevance]
    
    # Source filter
    if "allowed_sources" in filters:
        allowed_sources = filters["allowed_sources"]
        filtered = [ctx for ctx in filtered if ctx["source"] in allowed_sources]
    
    return filtered


def rank_contexts(contexts: List[Dict[str, Any]], ranking_criteria: List[str]) -> List[Dict[str, Any]]:
    """Rank contexts based on multiple criteria."""
    
    def calculate_composite_score(ctx):
        score = 0.0
        
        for criterion in ranking_criteria:
            if criterion == "relevance":
                score += ctx["relevance_score"] * 0.4
            elif criterion == "recency":
                # More recent contexts get higher scores
                ctx_date = datetime.fromisoformat(ctx["created_at"].replace('Z', '+00:00'))
                days_old = (datetime.now() - ctx_date.replace(tzinfo=None)).days
                recency_score = max(0, 1 - (days_old / 30))  # Decay over 30 days
                score += recency_score * 0.3
            elif criterion == "source_priority":
                # Memory store has higher priority
                source_scores = {"memory_store": 0.2, "context_store": 0.1}
                score += source_scores.get(ctx["source"], 0)
        
        return score
    
    # Calculate composite scores and sort
    for ctx in contexts:
        ctx["composite_score"] = calculate_composite_score(ctx)
    
    return sorted(contexts, key=lambda x: x["composite_score"], reverse=True)


def synthesize_context_items(contexts: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    """Synthesize context items into final format."""
    
    if mode == "comprehensive":
        # Include all context with full details
        return {
            "synthesis_mode": "comprehensive",
            "context_items": contexts,
            "total_items": len(contexts),
            "summary": f"Comprehensive context with {len(contexts)} items from multiple sources."
        }
    
    elif mode == "focused":
        # Include top contexts with key information
        focused_items = []
        for ctx in contexts[:10]:  # Top 10 items
            focused_items.append({
                "content": ctx["content"],
                "relevance": ctx["relevance_score"],
                "source": ctx["source"]
            })
        
        return {
            "synthesis_mode": "focused",
            "context_items": focused_items,
            "total_items": len(focused_items),
            "summary": f"Focused context with top {len(focused_items)} most relevant items."
        }
    
    elif mode == "summarized":
        # Create a summary of key points
        key_points = []
        sources_used = set()
        
        for ctx in contexts[:5]:  # Top 5 items
            content = ctx["content"]
            if isinstance(content, dict):
                # Extract key information
                for key, value in content.items():
                    if isinstance(value, str) and len(value) > 20:
                        key_points.append(f"{key}: {value[:100]}...")
            sources_used.add(ctx["source"])
        
        return {
            "synthesis_mode": "summarized",
            "key_points": key_points[:10],  # Top 10 key points
            "sources_used": list(sources_used),
            "total_items_synthesized": len(contexts),
            "summary": f"Summary of key insights from {len(contexts)} context items."
        }
    
    else:
        return {
            "synthesis_mode": mode,
            "error": f"Unknown synthesis mode: {mode}",
            "context_items": contexts
        } 