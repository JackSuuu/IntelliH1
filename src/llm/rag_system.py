"""
RAG System for Physics-Aware Planning (Phase 3)
Integrates LLM with physics knowledge using vector search
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging

from llm.physics_kb import PhysicsKnowledgeBase

logger = logging.getLogger(__name__)

# Try to import RAG dependencies
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Install with: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("SentenceTransformers not available. Install with: pip install sentence-transformers")

try:
    from groq import Groq
    from config import API_KEY
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq not available. Check config.py and API key")


class PhysicsRAGSystem:
    """
    Retrieval-Augmented Generation system for physics-aware manipulation
    
    Workflow:
    1. User query: "How to grasp coffee mug?"
    2. Retrieve relevant physics knowledge from vector DB
    3. LLM generates action plan using retrieved context
    4. Return physics-grounded manipulation strategy
    """
    
    def __init__(self):
        self.kb = PhysicsKnowledgeBase()
        
        # Initialize embedding model
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✓ SentenceTransformer loaded")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
        
        # Initialize vector store
        self.vector_index = None
        self.documents = []
        
        if FAISS_AVAILABLE and self.embedding_model:
            self._build_vector_index()
        
        # Initialize LLM client
        self.llm_client = None
        if GROQ_AVAILABLE:
            try:
                self.llm_client = Groq(api_key=API_KEY)
                logger.info("✓ Groq LLM client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")
        
        logger.info("[PhysicsRAGSystem] Initialized")
        logger.info(f"  Vector Index: {'✓' if self.vector_index else '✗'}")
        logger.info(f"  LLM: {'✓' if self.llm_client else '✗'}")
    
    def _build_vector_index(self):
        """Build FAISS vector index from knowledge base"""
        logger.info("Building vector index from knowledge base...")
        
        # Prepare documents from KB
        self.documents = []
        
        # Add object documents
        for obj in self.kb.objects_db:
            doc = f"Object: {obj['name']}\n"
            doc += f"Type: {obj['type']}, Shape: {obj['shape']}, Material: {obj['material']}\n"
            doc += f"Weight: {obj['typical_weight']['min']}-{obj['typical_weight']['max']} kg\n"
            doc += f"Fragile: {obj['fragile']}, Friction: {obj['friction_coefficient']}\n"
            doc += f"Grasp points: {', '.join(obj['grasp_points'])}\n"
            doc += f"Notes: {obj.get('manipulation_notes', '')}\n"
            
            self.documents.append({
                'text': doc,
                'type': 'object',
                'metadata': obj
            })
        
        # Add action documents
        for action in self.kb.actions_db:
            doc = f"Action: {action['action']}\n"
            doc += f"Applies to shapes: {', '.join(action['object_shapes'])}\n"
            doc += f"Requirements: {action['requirements']}\n"
            doc += f"Steps: {' -> '.join(action['steps'])}\n"
            doc += f"Failure modes: {', '.join(action['failure_modes'])}\n"
            
            self.documents.append({
                'text': doc,
                'type': 'action',
                'metadata': action
            })
        
        # Add failure mode documents
        for failure in self.kb.failure_modes_db:
            doc = f"Failure: {failure['failure']}\n"
            doc += f"Causes: {', '.join(failure['causes'])}\n"
            doc += f"Solutions: {' | '.join(failure['solutions'])}\n"
            doc += f"Prevention: {failure['prevention']}\n"
            
            self.documents.append({
                'text': doc,
                'type': 'failure_mode',
                'metadata': failure
            })
        
        # Embed all documents
        doc_texts = [d['text'] for d in self.documents]
        embeddings = self.embedding_model.encode(doc_texts, convert_to_numpy=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.vector_index = faiss.IndexFlatL2(dimension)
        self.vector_index.add(embeddings.astype('float32'))
        
        logger.info(f"✓ Vector index built with {len(self.documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Natural language query
            top_k: Number of documents to retrieve
        
        Returns:
            List of relevant documents with scores
        """
        if not self.vector_index or not self.embedding_model:
            logger.warning("Vector index not available, using fallback")
            return self._fallback_retrieve(query, top_k)
        
        try:
            # Embed query
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            
            # Search
            distances, indices = self.vector_index.search(
                query_embedding.astype('float32'), top_k
            )
            
            # Gather results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    results.append({
                        'document': self.documents[idx],
                        'score': float(distances[0][i]),
                        'rank': i + 1
                    })
            
            return results
        
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return self._fallback_retrieve(query, top_k)
    
    def _fallback_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Simple keyword-based retrieval fallback"""
        query_lower = query.lower()
        results = []
        
        for doc in self.documents:
            # Simple scoring based on keyword overlap
            text_lower = doc['text'].lower()
            score = sum(1 for word in query_lower.split() if word in text_lower)
            
            if score > 0:
                results.append({
                    'document': doc,
                    'score': -score,  # Negative for sorting (lower is better)
                    'rank': 0
                })
        
        # Sort by score
        results.sort(key=lambda x: x['score'])
        results = results[:top_k]
        
        # Update ranks
        for i, r in enumerate(results):
            r['rank'] = i + 1
        
        return results
    
    def generate_plan(self, query: str, scene_context: Optional[str] = None) -> str:
        """
        Generate manipulation plan using RAG
        
        Args:
            query: User query (e.g., "How to grasp coffee mug?")
            scene_context: Optional scene description for context
        
        Returns:
            LLM-generated manipulation plan
        """
        # Retrieve relevant knowledge
        retrieved_docs = self.retrieve(query, top_k=3)
        
        # Build context from retrieved documents
        context = "=== RELEVANT PHYSICS KNOWLEDGE ===\n\n"
        for result in retrieved_docs:
            context += f"[Document {result['rank']}]:\n"
            context += result['document']['text']
            context += "\n---\n\n"
        
        # Add scene context if provided
        if scene_context:
            context += f"=== CURRENT SCENE ===\n{scene_context}\n\n"
        
        # Build prompt
        prompt = f"""{context}

=== TASK ===
{query}

Based on the physics knowledge above, provide a detailed, step-by-step manipulation plan. Include:
1. Object analysis (shape, weight, fragility)
2. Recommended grasp strategy
3. Force requirements
4. Failure risks and mitigation
5. Step-by-step execution plan

Be specific and reference the physics properties."""
        
        # Generate with LLM
        if self.llm_client:
            try:
                response = self.llm_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a robotics expert specializing in physics-aware manipulation. "
                                      "You always ground your plans in physical properties and safety constraints."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                
                plan = response.choices[0].message.content
                return plan
            
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                return self._generate_plan_fallback(query, retrieved_docs)
        
        else:
            return self._generate_plan_fallback(query, retrieved_docs)
    
    def _generate_plan_fallback(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Fallback plan generation without LLM"""
        plan = f"Manipulation Plan for: {query}\n\n"
        plan += "Retrieved Knowledge:\n"
        
        for result in retrieved_docs:
            plan += f"\n{result['document']['text']}\n"
        
        plan += "\nNote: LLM not available. Showing retrieved knowledge only.\n"
        plan += "For full plan generation, ensure Groq API is configured.\n"
        
        return plan
    
    def query_and_plan(self, object_name: str, action: str, 
                       scene_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Complete RAG pipeline: Query KB + Generate Plan
        
        Args:
            object_name: Name of object to manipulate
            action: Action to perform (grasp, place, push, etc.)
            scene_info: Optional scene information
        
        Returns:
            Complete manipulation plan with physics grounding
        """
        # Build query
        query = f"How to {action} {object_name}?"
        
        # Get scene context
        scene_context = None
        if scene_info:
            scene_context = f"Scene: {scene_info.get('description', '')}\n"
            if 'objects' in scene_info:
                scene_context += f"Nearby objects: {', '.join(scene_info['objects'])}\n"
        
        # Retrieve physics knowledge
        retrieved = self.retrieve(query, top_k=3)
        
        # Check direct KB queries for specific info
        obj_props = self.kb.query_object(object_name)
        action_props = self.kb.query_action(f"{action}_{obj_props['shape'].split('_')[0]}" if obj_props else action)
        
        # Predict failure risks
        risk_assessment = None
        if obj_props:
            risk_assessment = self.kb.predict_failure_risk(
                object_name,
                f"{action}_object",
                {"force": 10}  # Default force estimate
            )
        
        # Generate plan with LLM
        plan_text = self.generate_plan(query, scene_context)
        
        return {
            'query': query,
            'object_properties': obj_props,
            'action_requirements': action_props,
            'retrieved_knowledge': retrieved,
            'risk_assessment': risk_assessment,
            'generated_plan': plan_text,
            'success_probability': self._estimate_success_probability(risk_assessment)
        }
    
    def _estimate_success_probability(self, risk_assessment: Optional[Dict]) -> float:
        """Estimate success probability based on risk assessment"""
        if not risk_assessment:
            return 0.5  # Unknown
        
        risk_level = risk_assessment.get('risk_level', 'unknown')
        
        if risk_level == 'low':
            return 0.9
        elif risk_level == 'medium':
            return 0.6
        elif risk_level == 'high':
            return 0.3
        else:
            return 0.5


# Example usage
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("="*60)
    print("Testing Physics RAG System")
    print("="*60 + "\n")
    
    # Initialize RAG system
    rag = PhysicsRAGSystem()
    
    # Test query
    query = "How should I grasp a coffee mug?"
    print(f"Query: {query}\n")
    
    # Retrieve relevant knowledge
    print("Retrieving relevant knowledge...")
    results = rag.retrieve(query, top_k=3)
    
    for r in results:
        print(f"\n[Rank {r['rank']}, Score: {r['score']:.4f}]")
        print(r['document']['text'][:200] + "...")
    
    # Generate complete plan
    print("\n" + "="*60)
    print("Generating complete manipulation plan...")
    print("="*60 + "\n")
    
    plan_result = rag.query_and_plan(
        "coffee_mug",
        "grasp",
        scene_info={
            'description': 'Kitchen counter with multiple objects',
            'objects': ['apple', 'banana', 'water_bottle']
        }
    )
    
    print(f"Object Properties:")
    if plan_result['object_properties']:
        print(f"  Fragile: {plan_result['object_properties']['fragile']}")
        print(f"  Material: {plan_result['object_properties']['material']}")
    
    print(f"\nRisk Assessment:")
    if plan_result['risk_assessment']:
        print(f"  Risk Level: {plan_result['risk_assessment']['risk_level']}")
        print(f"  Recommendations: {plan_result['risk_assessment']['recommendations']}")
    
    print(f"\nSuccess Probability: {plan_result['success_probability']:.2%}")
    
    print(f"\n{'='*60}")
    print("Generated Plan:")
    print(f"{'='*60}")
    print(plan_result['generated_plan'])
