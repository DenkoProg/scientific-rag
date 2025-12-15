import time

from loguru import logger

from scientific_rag.application.query.query_processor import QueryProcessor
from scientific_rag.application.rag.llm_client import llm_client
from scientific_rag.application.rag.prompt_templates import RAGPrompts
from scientific_rag.application.reranking.cross_encoder import reranker
from scientific_rag.application.retrieval.hybrid_retriever import HybridRetriever
from scientific_rag.domain.documents import PaperChunk
from scientific_rag.domain.responses import RAGResponse
from scientific_rag.settings import settings


class RAGPipeline:
    def __init__(self):
        logger.info("Initializing RAG Pipeline...")

        self.query_processor = QueryProcessor(expand_to_n=3)
        self.retriever = HybridRetriever()

        self.reranker = reranker
        self.llm = llm_client

        logger.info("RAG Pipeline initialized.")

    def run(
        self,
        query: str,
        # Toggles
        use_self_query: bool = True,
        use_query_expansion: bool = True,
        use_bm25: bool = True,
        use_dense: bool = True,
        use_reranking: bool = True,
        # Parameters
        retrieval_top_k: int = settings.retrieval_top_k,
        rerank_top_k: int = settings.rerank_top_k,
    ) -> RAGResponse:
        start_time = time.time()
        logger.info(f"Starting pipeline for query: '{query}'")

        # --- Step 1: Query Processing ---
        processed_query = self.query_processor.process(
            query, use_expansion=use_query_expansion, extract_filters=use_self_query
        )

        final_filters = processed_query.filters
        queries_to_run = processed_query.all_queries()
        logger.debug(f"Running search for {len(queries_to_run)} queries: {queries_to_run}")

        # --- Step 2: Retrieval loop ---
        chunk_map: dict[str, PaperChunk] = {}

        for q_str in queries_to_run:
            results = self.retriever.search(
                query=q_str, k=retrieval_top_k, filters=final_filters, use_bm25=use_bm25, use_dense=use_dense
            )

            for chunk in results:
                if chunk.chunk_id not in chunk_map:
                    chunk_map[chunk.chunk_id] = chunk
                else:
                    if chunk.score and chunk_map[chunk.chunk_id].score:
                        if chunk.score > chunk_map[chunk.chunk_id].score:
                            chunk_map[chunk.chunk_id] = chunk

        unique_chunks = list(chunk_map.values())
        logger.info(f"Retrieved {len(unique_chunks)} unique chunks before reranking")

        # --- Step 3: Reranking ---
        if use_reranking and unique_chunks:
            ranked_chunks = self.reranker.rerank(
                query=processed_query.original, chunks=unique_chunks, top_k=rerank_top_k
            )
        else:
            unique_chunks.sort(key=lambda x: x.score or 0.0, reverse=True)
            ranked_chunks = unique_chunks[:rerank_top_k]

        # --- Step 4: Generation ---
        if not ranked_chunks:
            answer = "I could not find any relevant information in the provided papers to answer your question."
        else:
            full_prompt = RAGPrompts.generate_rag_prompt(query, ranked_chunks)

            answer = self.llm.generate(prompt=full_prompt, system_prompt=RAGPrompts.SYSTEM_PROMPT)

        execution_time = time.time() - start_time

        return RAGResponse(
            answer=answer,
            original_query=query,
            generated_query_variations=processed_query.variations,
            retrieved_chunks=ranked_chunks,
            used_filters=final_filters.model_dump() if final_filters else None,
            execution_time=execution_time,
        )
