"""
LangGraph Workflow Setup for ChaBo RAG Orchestrator
"""
import logging
from functools import partial
from typing import Dict, Optional
from langgraph.graph import StateGraph, START, END

from .state import GraphState
from .nodes import retrieve_node, generate_node_streaming, ingest_node, extract_filters_node, rewrite_query_node
from components.rewriter.db_context import DBContext

logger = logging.getLogger(__name__)


def build_workflow(
    retriever_instance,
    generator_instance,
    filterable_fields: Dict[str, str] = None,
    filter_values: Dict[str, list] = None,
    db_context: Optional["DBContext"] = None,
    rewriter_enabled: bool = False,
):
    """
    Build and compile the LangGraph workflow for RAG orchestration.

    Args:
        retriever_instance: Initialised ChaBoHFEndpointRetriever
        generator_instance: Initialised Generator
        filterable_fields: Dict of {field_name: type} for LLM-based metadata filter extraction.
                           Pass {} or None to disable (extract_filters node becomes a pass-through).
        filter_values: Dict of {field_name: [valid_values]} for constrained LLM extraction.
                       Every field in filterable_fields must have an entry here.
        db_context: Optional DBContext for query rewriter. Required if rewriter_enabled=True.
        rewriter_enabled: When True, inserts rewrite_query_node between ingest and extract_filters.
                          When False, the node is omitted and downstream nodes read state['query'] directly.
    """
    if filterable_fields is None:
        filterable_fields = {}
    if filter_values is None:
        filter_values = {}

    workflow = StateGraph(GraphState)

    # Inject services into nodes
    r_node = partial(retrieve_node, retriever=retriever_instance)
    g_node = partial(generate_node_streaming, generator=generator_instance)
    f_node = partial(extract_filters_node, generator=generator_instance, filterable_fields=filterable_fields, filter_values=filter_values)

    # Add nodes
    workflow.add_node("ingest", ingest_node)
    workflow.add_node("extract_filters", f_node)
    workflow.add_node("retrieve", r_node)
    workflow.add_node("generate", g_node)

    # Define edges
    workflow.add_edge(START, "ingest")

    if rewriter_enabled:
        if db_context is None:
            raise ValueError("build_workflow: rewriter_enabled=True requires a db_context")
        rw_node = partial(rewrite_query_node, generator=generator_instance, db_context=db_context)
        workflow.add_node("rewrite_query", rw_node)
        workflow.add_edge("ingest", "rewrite_query")
        workflow.add_edge("rewrite_query", "extract_filters")
    else:
        workflow.add_edge("ingest", "extract_filters")

    workflow.add_edge("extract_filters", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    compiled_graph = workflow.compile()

    logger.info(
        f"LangGraph workflow compiled "
        f"(filterable_fields={list(filterable_fields.keys())}, "
        f"filter_values_fields={list(filter_values.keys())}, "
        f"rewriter_enabled={rewriter_enabled})"
    )
    return compiled_graph