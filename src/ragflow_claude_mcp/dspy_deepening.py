"""
DSPy-powered query deepening for RAGFlow search optimization.

This module implements intelligent query refinement using DSPy to improve
search result quality through iterative query optimization.
"""

import dspy
from typing import Dict, List, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


class AnalyzeSearchResults(dspy.Signature):
    """Analyze search results to identify what information is missing or unclear"""
    
    original_query: str = dspy.InputField(desc="The user's original search query")
    search_results: str = dspy.InputField(desc="Summary of top search results from RAGFlow")
    result_count: int = dspy.InputField(desc="Number of results found")
    
    missing_aspects: List[str] = dspy.OutputField(desc="What aspects of the query aren't well covered in results")
    ambiguous_terms: List[str] = dspy.OutputField(desc="Terms that might need clarification or expansion")
    result_quality: str = dspy.OutputField(desc="Assessment of result quality: 'excellent', 'good', 'poor', or 'insufficient'")
    refinement_needed: bool = dspy.OutputField(desc="Whether query refinement would likely improve results")


class RefineQuery(dspy.Signature):
    """Generate a refined search query based on gap analysis"""
    
    original_query: str = dspy.InputField(desc="The user's original search query")
    missing_aspects: List[str] = dspy.InputField(desc="Information gaps identified in initial results")
    context_from_results: str = dspy.InputField(desc="Key themes and context from initial results")
    refinement_iteration: int = dspy.InputField(desc="Which iteration of refinement (1, 2, 3, etc.)")
    
    refined_query: str = dspy.OutputField(desc="Improved query targeting missing information")
    refinement_reasoning: str = dspy.OutputField(desc="Explanation of why this refinement should work better")
    expected_improvement: str = dspy.OutputField(desc="What specific improvements are expected")


class MergeSearchResults(dspy.Signature):
    """Intelligently merge and rank results from multiple query iterations"""
    
    original_query: str = dspy.InputField(desc="The user's original search query")
    query_iterations: List[str] = dspy.InputField(desc="All queries used in the search process")
    all_results_summary: str = dspy.InputField(desc="Summary of all search results")
    max_results: int = dspy.InputField(desc="Maximum number of results to return")
    
    merge_strategy: str = dspy.OutputField(desc="Strategy used for merging results")
    quality_ranking: List[str] = dspy.OutputField(desc="Ranking criteria used for final results")
    final_result_summary: str = dspy.OutputField(desc="Summary of merged results quality")


class DSPyQueryDeepener(dspy.Module):
    """DSPy module for iterative query refinement and result optimization"""
    
    def __init__(self):
        super().__init__()
        self.analyzer = dspy.ChainOfThought(AnalyzeSearchResults)
        self.refiner = dspy.ChainOfThought(RefineQuery)
        self.merger = dspy.ChainOfThought(MergeSearchResults)
        
    def extract_content_summary(self, results: Dict[str, Any]) -> str:
        """Extract key content from RAGFlow results for analysis"""
        if not results or not results.get('data'):
            return "No results found"
        
        data = results['data']
        if not data:
            return "Empty results"
        
        # Extract content from top results
        content_snippets = []
        for i, result in enumerate(data[:5]):  # Top 5 results
            content = result.get('content_with_weight', result.get('content', ''))
            if content:
                # Take meaningful snippet
                snippet = content[:300] + "..." if len(content) > 300 else content
                content_snippets.append(f"Result {i+1}: {snippet}")
        
        if not content_snippets:
            return "No content found in results"
        
        return "\n---\n".join(content_snippets)
    
    def extract_key_context(self, results: Dict[str, Any]) -> str:
        """Extract key themes and context from results"""
        if not results or not results.get('data'):
            return "No context available"
        
        data = results['data']
        if not data:
            return "No context available"
        
        # Extract key terms and themes
        all_content = []
        for result in data[:3]:  # Top 3 for context
            content = result.get('content_with_weight', result.get('content', ''))
            if content:
                all_content.append(content)
        
        if not all_content:
            return "No context available"
        
        # Simple context extraction - in a real system, this could be more sophisticated
        combined_content = " ".join(all_content)
        # Take first 500 characters as context
        context = combined_content[:500]
        
        return context
    
    async def deepen_search(self, 
                           ragflow_client, 
                           dataset_ids: List[str], 
                           original_query: str, 
                           deepening_level: int = 1,
                           **ragflow_kwargs) -> Dict[str, Any]:
        """
        Perform iterative query deepening to improve search results
        
        Args:
            ragflow_client: RAGFlow client instance
            dataset_ids: List of Dataset IDs to search
            original_query: User's original search query
            deepening_level: Number of refinement iterations (1-3)
            **ragflow_kwargs: Additional arguments for RAGFlow search
            
        Returns:
            Dictionary containing refined results and process information
        """
        
        if deepening_level <= 0:
            # No deepening, just return original results
            results = await ragflow_client.retrieval_query(
                dataset_ids=dataset_ids,
                query=original_query,
                **ragflow_kwargs
            )
            return {
                "original_query": original_query,
                "final_query": original_query,
                "queries_used": [original_query],
                "deepening_level": 0,
                "results": results,
                "refinement_log": []
            }
        
        # Track the deepening process
        queries_used = [original_query]
        all_results = []
        refinement_log = []
        
        # Step 1: Initial search
        logger.info(f"Starting search deepening for query: '{original_query}'")
        current_results = await ragflow_client.retrieval_query(
            dataset_ids=dataset_ids,
            query=original_query,
            **ragflow_kwargs
        )
        
        all_results.append({
            "query": original_query,
            "results": current_results,
            "iteration": 0
        })
        
        current_query = original_query
        
        # Iterative refinement
        for iteration in range(1, deepening_level + 1):
            logger.info(f"Starting refinement iteration {iteration}")
            
            # Analyze current results
            content_summary = self.extract_content_summary(current_results)
            result_count = len(current_results.get('data', []))
            
            try:
                analysis = self.analyzer(
                    original_query=original_query,
                    search_results=content_summary,
                    result_count=result_count
                )
                
                # Log analysis
                refinement_log.append({
                    "iteration": iteration,
                    "stage": "analysis",
                    "result_quality": analysis.result_quality,
                    "missing_aspects": analysis.missing_aspects,
                    "refinement_needed": analysis.refinement_needed
                })
                
                # If refinement not needed, stop early
                if not analysis.refinement_needed:
                    logger.info(f"Refinement not needed at iteration {iteration}, stopping")
                    break
                
                # Generate refined query
                context = self.extract_key_context(current_results)
                
                refinement = self.refiner(
                    original_query=original_query,
                    missing_aspects=analysis.missing_aspects,
                    context_from_results=context,
                    refinement_iteration=iteration
                )
                
                refined_query = refinement.refined_query
                queries_used.append(refined_query)
                
                # Log refinement
                refinement_log.append({
                    "iteration": iteration,
                    "stage": "refinement",
                    "refined_query": refined_query,
                    "reasoning": refinement.refinement_reasoning,
                    "expected_improvement": refinement.expected_improvement
                })
                
                # Execute refined search
                logger.info(f"Executing refined search: '{refined_query}'")
                refined_results = await ragflow_client.retrieval_query(
                    dataset_ids=dataset_ids,
                    query=refined_query,
                    **ragflow_kwargs
                )
                
                all_results.append({
                    "query": refined_query,
                    "results": refined_results,
                    "iteration": iteration
                })
                
                current_results = refined_results
                current_query = refined_query
                
            except Exception as e:
                logger.error(f"Error in refinement iteration {iteration}: {e}")
                # Continue with current results if refinement fails
                break
        
        # Merge results if we have multiple iterations
        final_results = current_results
        merge_info = None
        
        if len(all_results) > 1:
            try:
                # Prepare data for merging
                query_iterations = [r["query"] for r in all_results]
                all_results_summary = self._create_results_summary(all_results)
                
                merge_analysis = self.merger(
                    original_query=original_query,
                    query_iterations=query_iterations,
                    all_results_summary=all_results_summary,
                    max_results=ragflow_kwargs.get('page_size', 10)
                )
                
                merge_info = {
                    "strategy": merge_analysis.merge_strategy,
                    "quality_ranking": merge_analysis.quality_ranking,
                    "summary": merge_analysis.final_result_summary
                }
                
                # For now, use the last (most refined) results
                # In a more sophisticated implementation, we'd actually merge
                final_results = current_results
                
            except Exception as e:
                logger.error(f"Error in result merging: {e}")
                # Fall back to last results
                final_results = current_results
        
        logger.info(f"Search deepening completed. Used {len(queries_used)} queries.")
        
        return {
            "original_query": original_query,
            "final_query": current_query,
            "queries_used": queries_used,
            "deepening_level": deepening_level,
            "results": final_results,
            "refinement_log": refinement_log,
            "merge_info": merge_info,
            "all_results": all_results
        }
    
    def _create_results_summary(self, all_results: List[Dict]) -> str:
        """Create a summary of all search results for merging analysis"""
        summary_parts = []
        
        for result_set in all_results:
            query = result_set["query"]
            results = result_set["results"]
            iteration = result_set["iteration"]
            
            result_count = len(results.get('data', []))
            avg_similarity = 0
            
            if results.get('data'):
                similarities = [r.get('similarity', 0) for r in results['data']]
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            
            summary_parts.append(
                f"Iteration {iteration} - Query: '{query}' - "
                f"Results: {result_count} - Avg Similarity: {avg_similarity:.3f}"
            )
        
        return "\n".join(summary_parts)


# Global instance for reuse
_deepener_instance = None

def get_deepener() -> DSPyQueryDeepener:
    """Get or create a global DSPy query deepener instance"""
    global _deepener_instance
    if _deepener_instance is None:
        _deepener_instance = DSPyQueryDeepener()
    return _deepener_instance