# data_loader.py
import os
import json
import logging
from typing import List, Dict, Any, Optional, Literal, Tuple
from pathlib import Path
import warnings

# --- Required Libraries ---
# Ensure these are installed:
# pip install langchain langchain-community langchain-google-genai faiss-cpu sentence-transformers google-generativeai python-dotenv rank_bm25 transformers>=4.34.0

# --- Langchain Core Imports ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# --- Langchain Community Imports ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Optional Provider Imports ---
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    import google.generativeai as genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    # No warning here, will be handled during initialization if 'google' provider is chosen

# --- Retriever Imports ---
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# --- Environment and Setup ---
from dotenv import load_dotenv

# Ensure rank_bm25 is installed for BM25Retriever
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("BM25Retriever requires the 'rank_bm25' library. Please install it using: pip install rank_bm25")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables (especially for GOOGLE_API_KEY)
load_dotenv()

# --- Configuration ---
DEFAULT_HUGGINGFACE_MODEL = "all-MiniLM-L6-v2"
# <<< CHANGE: Updated default Google model to text-embedding-004 >>>
DEFAULT_GOOGLE_MODEL = "models/text-embedding-004"

# --- Advanced Data Loader with Hybrid Search ---

class CricketDataLoaderV2:
    """
    An advanced data loader for cricket JSON data, supporting multiple embedding providers,
    intelligent document creation, persistence, and hybrid retrieval (FAISS + BM25).
    Specifically configured to support Google's 'text-embedding-004'.
    """
    def __init__(
        self,
        data_dir: str,
        persist_dir: str = "./cricket_faiss_index",
        embedding_provider: Literal['huggingface', 'google'] = 'huggingface',
        model_name: Optional[str] = None, # Can override the default model
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        verbose: bool = True
    ):
        """
        Initializes the CricketDataLoaderV2.

        Args:
            data_dir (str): Path to the directory containing JSON data files.
            persist_dir (str): Directory to save/load the FAISS index.
            embedding_provider (Literal['huggingface', 'google']): The embedding provider for FAISS.
            model_name (Optional[str]): Specific model name for the chosen provider.
                                        If 'google' provider and None, defaults to DEFAULT_GOOGLE_MODEL.
            chunk_size (int): Maximum size of text chunks.
            chunk_overlap (int): Overlap between text chunks.
            verbose (bool): If True, enables detailed logging.
        """
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)
        self.embedding_provider = embedding_provider
        # Store the effective model name used for status reporting
        self._effective_model_name = model_name # User-provided overrides default
        if self.embedding_provider == 'google' and not self._effective_model_name:
            self._effective_model_name = DEFAULT_GOOGLE_MODEL
        elif self.embedding_provider == 'huggingface' and not self._effective_model_name:
            self._effective_model_name = DEFAULT_HUGGINGFACE_MODEL
        # Use the originally passed model_name (could be None) for initialization logic below
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if not verbose:
            logger.setLevel(logging.WARNING)
        else:
             logger.setLevel(logging.INFO) # Ensure INFO level if verbose

        if not self.data_dir.is_dir():
            logger.error(f"Data directory not found: {self.data_dir}")
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.embeddings = self._initialize_embeddings() # This now uses self.model_name or defaults
        self.vector_store: Optional[FAISS] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.ensemble_retriever: Optional[EnsembleRetriever] = None
        self.all_split_texts: List[Document] = [] # Store split texts for BM25 index

        # Loading status trackers
        self.loaded_files_info: Dict[str, str] = {} # filename: status
        self.processed_items_count = 0
        self.failed_items_count = 0

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""], # Common separators
            add_start_index=True, # Useful for potential context linking
        )

        logger.info(f"CricketDataLoaderV2 initialized.")
        logger.info(f"Embedding Provider: {self.embedding_provider}")
        if self.embeddings:
            logger.info(f"Effective Embedding Model: {self._effective_model_name}")
        logger.info(f"FAISS Persistence Directory: {self.persist_dir}")

    def _initialize_embeddings(self):
        """Initializes the embedding model based on the selected provider."""
        logger.info(f"Initializing embeddings for provider: '{self.embedding_provider}'...")
        try:
            if self.embedding_provider == 'huggingface':
                # Use provided model_name or the default HF model
                model_to_use = self.model_name or DEFAULT_HUGGINGFACE_MODEL
                logger.info(f"Using HuggingFace model: {model_to_use}")
                # Ensure trust_remote_code is handled appropriately based on model requirements
                # For standard models like all-MiniLM-L6-v2, it's usually not needed.
                # Set trust_remote_code=True only if using models that require it.
                hf_kwargs = {'trust_remote_code': True} if "instructor" in model_to_use else {}
                return HuggingFaceEmbeddings(model_name=model_to_use, model_kwargs=hf_kwargs)

            elif self.embedding_provider == 'google':
                if not GOOGLE_GENAI_AVAILABLE:
                     logger.error("Google Generative AI libraries not found. Install 'langchain-google-genai' and 'google-generativeai'.")
                     raise RuntimeError("Google GenAI libraries not installed. Cannot use 'google' provider.")

                google_api_key = os.getenv("GOOGLE_API_KEY")
                if not google_api_key:
                    logger.error("GOOGLE_API_KEY environment variable not set. Required for 'google' provider.")
                    raise ValueError("GOOGLE_API_KEY environment variable not set for 'google' provider.")

                try:
                    # Configure genai if not already done (idempotent)
                    genai.configure(api_key=google_api_key)
                except Exception as config_err:
                     logger.error(f"Failed to configure Google GenAI with API key: {config_err}", exc_info=True)
                     raise ValueError(f"Failed to configure Google GenAI API: {config_err}")

                # <<< CHANGE: Use provided model_name or the new default Google model >>>
                model_to_use = self.model_name or DEFAULT_GOOGLE_MODEL
                logger.info(f"Using Google GenAI embedding model: {model_to_use}")

                # Check if the model exists (optional but recommended)
                try:
                     genai.get_model(model_to_use) # Throws if model doesn't exist or isn't accessible
                     logger.info(f"Verified model '{model_to_use}' is available via Google API.")
                except Exception as model_check_err:
                     logger.error(f"Failed to verify Google embedding model '{model_to_use}': {model_check_err}. Ensure the model name is correct and accessible with your API key.")
                     # Depending on strictness, you might raise an error here or proceed cautiously
                     # raise ValueError(f"Google embedding model '{model_to_use}' not accessible: {model_check_err}")

                return GoogleGenerativeAIEmbeddings(model=model_to_use)
            else:
                # This should not happen due to Literal typing, but good practice
                raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}", exc_info=True)
            raise # Re-raise the exception after logging

    def _extract_meaningful_content(self, item: Dict[str, Any]) -> str:
        """
        Tries to extract a primary text field from a JSON object.
        Falls back to JSON dump if no specific field is found.
        """
        potential_keys = ['commentary', 'text', 'summary', 'description', 'content', 'details', 'innings_summary', 'match_report', 'body']
        for key in potential_keys:
            content = item.get(key)
            if isinstance(content, str) and content.strip():
                logger.debug(f"Extracted content from key: '{key}'")
                return content.strip()
            elif isinstance(content, list): # Handle lists of strings (e.g., multiple paragraphs)
                 str_content = "\n".join(filter(lambda x: isinstance(x, str), content))
                 if str_content.strip():
                     logger.debug(f"Extracted content from list in key: '{key}'")
                     return str_content.strip()

        # Fallback: Pretty-print JSON, maybe excluding large non-textual fields
        logger.debug("No primary text field found, attempting fallback to JSON dump of simple values.")
        try:
            # Filter out complex types or potentially very large fields before dumping
            keys_to_include = {k: v for k, v in item.items() if isinstance(v, (str, int, float, bool)) and len(str(v)) < 500} # More selective filter
            if keys_to_include:
                 return json.dumps(keys_to_include, indent=2)
            else:
                 logger.warning(f"No primary text field or simple values found for fallback JSON dump. Keys present: {list(item.keys())}. Returning empty string.")
                 return "" # Avoid dumping large complex objects if no suitable content found
        except Exception as e:
            logger.error(f"Error during fallback JSON dump: {e}")
            return ""

    def _create_documents(self, all_data: List[Dict[str, Any]], filename: str) -> List[Document]:
        """Converts raw JSON data items into Langchain Document objects with cleaned metadata."""
        documents = []
        processed_count_file = 0
        failed_count_file = 0
        for index, item in enumerate(all_data):
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dict item at index {index} in file {filename}")
                failed_count_file += 1
                continue

            try:
                content = self._extract_meaningful_content(item)
                if not content:
                    logger.warning(f"Skipping item at index {index} in file {filename} due to empty or non-extractable content.")
                    failed_count_file += 1
                    continue

                # --- Enhanced Metadata Extraction ---
                metadata = {
                    "source_file": filename,
                    "item_index": index,
                    # Attempt to get specific identifiers first
                    "match_id": str(item.get("match_id", item.get("matchId", item.get("id", "")))).strip(),
                    "data_type": str(item.get("data_type", item.get("type", "unknown"))).strip().lower(),
                    "match_date": str(item.get("match_date", item.get("date", ""))).strip(),
                    "venue": str(item.get("venue", item.get("location", ""))).strip(),
                    "inning": str(item.get("inning", item.get("innings", ""))).strip(),
                    # Try to get teams as a list first, then convert to string
                    "teams": item.get("teams", []),
                    "player": str(item.get("player", item.get("batsman", item.get("bowler", "")))).strip(), # Combine common player fields
                    # Add other potentially useful fields
                    "result": str(item.get("result", "")).strip(),
                }

                # --- Metadata Cleaning ---
                # Convert list of teams to a comma-separated string if it exists
                if isinstance(metadata["teams"], list) and metadata["teams"]:
                     metadata["teams"] = ", ".join(map(str, metadata["teams"]))
                elif not isinstance(metadata["teams"], str): # Handle cases where it might be something else
                     metadata["teams"] = ""


                # Remove empty values AFTER potential extraction
                metadata = {k: v for k, v in metadata.items() if v not in [None, "", []]}

                documents.append(Document(page_content=content, metadata=metadata))
                processed_count_file += 1
            except Exception as e:
                logger.error(f"Error processing item at index {index} in {filename}: {e}", exc_info=False) # Less verbose logging for item errors
                failed_count_file += 1

        # Update global counts
        self.processed_items_count += processed_count_file
        self.failed_items_count += failed_count_file
        logger.debug(f"File {filename}: Created {processed_count_file} documents, failed/skipped {failed_count_file} items.")
        return documents

    def _load_process_and_split_docs(self) -> Tuple[List[Document], List[Document]]:
        """Loads JSON files, processes items into documents, and splits them."""
        self.loaded_files_info = {} # Reset status for this run
        self.processed_items_count = 0 # Reset counts for this run
        self.failed_items_count = 0
        all_documents = []
        json_files = list(self.data_dir.glob('*.json')) + list(self.data_dir.glob('*.jsonl')) # Include .jsonl
        logger.info(f"Found {len(json_files)} JSON/JSONL files in {self.data_dir}")

        if not json_files:
            logger.warning(f"No JSON or JSONL files found in {self.data_dir}.")
            return [], []

        for filepath in json_files:
            filename = filepath.name
            logger.info(f"Processing file: {filename}") # Log each file being processed
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip():
                        logger.warning(f"File {filename} is empty. Skipping.")
                        self.loaded_files_info[filename] = "skipped (empty)"
                        continue

                    if filepath.suffix == '.jsonl':
                        logger.debug(f"Processing {filename} as JSON Lines format.")
                        try:
                            raw_data = [json.loads(line) for line in content.strip().split('\n') if line.strip()]
                        except json.JSONDecodeError as e:
                            logger.error(f"Error decoding JSON Line in {filename}: {e}. Skipping file.")
                            self.loaded_files_info[filename] = f"failed: JSONDecodeError in line"
                            continue
                    else: # Assume .json
                        logger.debug(f"Processing {filename} as standard JSON format.")
                        try:
                            raw_data = json.loads(content)
                        except json.JSONDecodeError as e:
                             logger.error(f"Error decoding JSON in {filename}: {e}. Skipping file.")
                             self.loaded_files_info[filename] = f"failed: JSONDecodeError"
                             continue

                # Handle different data structures after loading
                if isinstance(raw_data, dict):
                    # If it's a dict, maybe it contains a list under a key like 'data' or 'records'
                    potential_list_keys = ['data', 'records', 'items', 'matches', 'commentary_items']
                    data_list = None
                    for key in potential_list_keys:
                        if isinstance(raw_data.get(key), list):
                            data_list = raw_data[key]
                            logger.debug(f"Found data list under key '{key}' in {filename}")
                            break
                    if data_list is None: # Treat the single dict as the only item
                        data_list = [raw_data]
                        logger.debug(f"Treating single dictionary in {filename} as one data item.")

                elif isinstance(raw_data, list):
                    data_list = raw_data
                else:
                    logger.warning(f"Skipping file {filename}: Content is not a recognized JSON structure (dict or list). Type: {type(raw_data)}")
                    self.loaded_files_info[filename] = f"failed: Invalid JSON structure type"
                    continue

                # Ensure we have a list of dictionaries
                if not isinstance(data_list, list) or not all(isinstance(item, dict) for item in data_list):
                     logger.warning(f"Skipping file {filename}: Final data is not a list of dictionaries.")
                     self.loaded_files_info[filename] = f"failed: Content not list of dicts"
                     continue

                if not data_list:
                     logger.warning(f"No data items found in {filename} after parsing.")
                     self.loaded_files_info[filename] = "loaded (0 items)"
                     continue

                # --- Process into Documents ---
                file_docs = self._create_documents(data_list, filename)
                if file_docs:
                    all_documents.extend(file_docs)
                    self.loaded_files_info[filename] = "loaded"
                    logger.info(f"Successfully processed {filename} -> {len(file_docs)} documents generated.")
                else:
                    # Handled within _create_documents if no docs are made from items
                    if filename not in self.loaded_files_info: # Only log if not already marked failed
                         logger.warning(f"No documents generated from {filename}, though file was read.")
                         self.loaded_files_info[filename] = "loaded (0 docs created)"

            except Exception as e:
                logger.error(f"Critical error processing file {filename}: {e}", exc_info=True)
                self.loaded_files_info[filename] = f"failed: {type(e).__name__}"

        if not all_documents:
            logger.error("No documents were successfully created from any file.")
            return [], []

        # --- Splitting ---
        logger.info(f"Total documents created across all files: {len(all_documents)}")
        logger.info(f"Splitting {len(all_documents)} documents using chunk size {self.chunk_size}, overlap {self.chunk_overlap}...")
        try:
             split_texts = self.text_splitter.split_documents(all_documents)
             logger.info(f"Successfully split into {len(split_texts)} text chunks.")
             self.all_split_texts = split_texts # Store for BM25 and status
        except Exception as e:
             logger.error(f"Error during document splitting: {e}", exc_info=True)
             return all_documents, [] # Return original docs but empty splits


        return all_documents, self.all_split_texts

    def _initialize_bm25_retriever(self, texts: List[Document]):
        """Initializes the BM25 retriever from split documents."""
        if not texts:
             logger.warning("No text chunks provided to initialize BM25 retriever. Skipping BM25 setup.")
             self.bm25_retriever = None
             return

        logger.info(f"Initializing BM25 retriever from {len(texts)} text chunks...")
        try:
            # BM25Retriever automatically extracts 'page_content'
            self.bm25_retriever = BM25Retriever.from_documents(
                documents=texts
                # parameters: k1=1.5, b=0.75 are defaults, can be tuned
            )
            # Set default k for BM25 retrieval if desired
            self.bm25_retriever.k = 5 # Example: Default to retrieving top 5 BM25 results
            logger.info("BM25 retriever initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize BM25 retriever: {e}", exc_info=True)
            self.bm25_retriever = None

    def _initialize_ensemble_retriever(self):
        """Initializes the Ensemble retriever combining FAISS and BM25."""
        if not self.vector_store:
            logger.warning("FAISS vector store is not initialized. Cannot create Ensemble Retriever.")
            self.ensemble_retriever = None
            return
        if not self.bm25_retriever:
            logger.warning("BM25 retriever is not initialized. Cannot create Ensemble Retriever.")
            # Could proceed with only FAISS, but ensemble implies both are intended
            self.ensemble_retriever = None
            return

        logger.info("Initializing Ensemble retriever...")
        # Define the base retrievers
        # Adjust 'k' here based on how many results you want from each *before* ensembling
        faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        # BM25Retriever's k was set during its initialization

        # Define weights - IMPORTANT: TUNE THESE BASED ON EXPERIMENTS
        # Higher weight means more influence on the final ranking.
        # Example: Give slightly more weight to semantic search (FAISS)
        ensemble_weights = [0.4, 0.6] # Example: 40% BM25 (keyword), 60% FAISS (semantic)
        logger.info(f"Using Ensemble weights: BM25={ensemble_weights[0]}, FAISS={ensemble_weights[1]}")

        try:
             self.ensemble_retriever = EnsembleRetriever(
                 retrievers=[self.bm25_retriever, faiss_retriever],
                 weights=ensemble_weights
                 # Optional: search_type="mmr" for Maximum Marginal Relevance can improve diversity
                 # search_type="mmr", search_kwargs={'k': 6, 'fetch_k': 20, 'lambda_mult': 0.6} # Example MMR config
             )
             logger.info("Ensemble retriever initialized successfully.")
        except Exception as e:
             logger.error(f"Failed to initialize Ensemble retriever: {e}", exc_info=True)
             self.ensemble_retriever = None


    def load_and_process_data(self, force_rebuild: bool = False):
        """
        Loads data, processes it, builds/loads FAISS index, builds BM25 index,
        and creates the ensemble retriever. Ensures embeddings are initialized.

        Args:
            force_rebuild (bool): If True, ignores existing FAISS index and rebuilds everything.

        Returns:
            Optional[FAISS]: The FAISS vector store instance, or None if initialization failed.
        """
        # Ensure embeddings are ready before proceeding
        if not self.embeddings:
             logger.error("Embeddings were not initialized successfully. Cannot load or process data.")
             return None

        faiss_index_file = self.persist_dir / "index.faiss"
        faiss_pkl_file = self.persist_dir / "index.pkl"

        # --- Attempt to load FAISS ---
        if not force_rebuild and faiss_index_file.exists() and faiss_pkl_file.exists():
            logger.info(f"Attempting to load existing FAISS vector store from {self.persist_dir}...")
            try:
                # Determine if dangerous deserialization is needed (primarily for custom HF models)
                allow_dangerous = self.embedding_provider == 'huggingface'
                # Suppress specific warnings during loading if necessary
                # with warnings.catch_warnings():
                #      warnings.simplefilter("ignore", category=UserWarning) # Example
                self.vector_store = FAISS.load_local(
                    folder_path=str(self.persist_dir),
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=allow_dangerous
                )
                logger.info(f"Successfully loaded FAISS vector store. Contains {self.vector_store.index.ntotal} vectors.")

                # --- Load documents and build BM25 + Ensemble ---
                # We still need the documents in memory for BM25, even if FAISS is loaded.
                # This also updates the file loading status.
                logger.info("Loading documents to build BM25 index (even though FAISS loaded)...")
                _, self.all_split_texts = self._load_process_and_split_docs() # Re-run load/split
                if not self.all_split_texts:
                     logger.error("Failed to load/split documents needed for BM25 index. Hybrid search disabled.")
                     # Keep the loaded FAISS store but disable ensemble
                     self.bm25_retriever = None
                     self.ensemble_retriever = None
                else:
                     self._initialize_bm25_retriever(self.all_split_texts)
                     self._initialize_ensemble_retriever() # Attempt ensemble setup

                return self.vector_store # Return loaded store

            except Exception as e:
                logger.warning(f"Failed to load FAISS store from {self.persist_dir}: {e}. Proceeding to rebuild...", exc_info=True)
                self.vector_store = None # Reset store
                self.all_split_texts = [] # Reset texts

        # --- Build FAISS and BM25 from scratch ---
        logger.info(f"Building index from scratch (force_rebuild={force_rebuild} or load failed)...")

        _, self.all_split_texts = self._load_process_and_split_docs() # Load and split fresh

        if not self.all_split_texts:
            logger.error("No text chunks were generated after loading and splitting. Cannot build FAISS or BM25.")
            self.vector_store = None
            self.bm25_retriever = None
            self.ensemble_retriever = None
            return None

        # --- Create FAISS ---
        logger.info("Creating new FAISS vector store...")
        try:
            self.vector_store = FAISS.from_documents(self.all_split_texts, self.embeddings)
            logger.info(f"FAISS vector store created with {self.vector_store.index.ntotal} vectors.")

            # --- Save FAISS ---
            logger.info(f"Saving FAISS vector store to {self.persist_dir}...")
            self.vector_store.save_local(str(self.persist_dir))
            logger.info("FAISS vector store saved successfully.")
        except Exception as e:
            logger.error(f"Failed to create or save FAISS index: {e}", exc_info=True)
            self.vector_store = None # Ensure inconsistent state isn't kept

        # --- Create BM25 & Ensemble (needs FAISS) ---
        self._initialize_bm25_retriever(self.all_split_texts)
        if self.vector_store: # Only init ensemble if FAISS was successful
             self._initialize_ensemble_retriever()
        else:
             logger.warning("FAISS store failed to initialize, cannot create ensemble retriever.")
             self.ensemble_retriever = None # Ensure it's None


        return self.vector_store

    def get_relevant_context(self, query: str, k: int = 5, use_hybrid: bool = True) -> str:
        """
        Retrieves relevant context chunks. Prefers the ENSEMBLE retriever if available and use_hybrid=True.
        Falls back to FAISS, then BM25 if needed.

        Args:
            query (str): The user's query.
            k (int): The desired maximum number of final context chunks.
            use_hybrid (bool): If True and ensemble retriever exists, use it. Otherwise, fallback.

        Returns:
            str: A formatted string containing the relevant contexts.
                 Returns an error message if no retriever is available or successful.
        """
        logger.info(f"Retrieving context for query: '{query}' (k={k}, use_hybrid={use_hybrid})")
        retriever_used = "None"
        documents: List[Document] = []
        final_documents: List[Document] = []


        # 1. Try Ensemble Retriever (if enabled and available)
        if use_hybrid and self.ensemble_retriever:
            retriever_used = "Ensemble (BM25 + FAISS)"
            logger.debug(f"Attempting search with {retriever_used}...")
            try:
                # Ensemble retriever doesn't directly take 'k', it depends on underlying retrievers and ranking
                # We retrieve potentially more and then limit to 'k' later.
                documents = self.ensemble_retriever.get_relevant_documents(query)
                logger.debug(f"Ensemble search returned {len(documents)} documents.")
            except Exception as e:
                logger.error(f"Error during ensemble search: {e}", exc_info=True)
                documents = [] # Reset documents list on error

        # 2. Try FAISS (if Ensemble not used/failed or returned no results)
        if not documents and self.vector_store:
             fallback_reason = "Ensemble not used or yielded no results" if use_hybrid else "Hybrid search disabled"
             logger.info(f"{fallback_reason}. Falling back to FAISS search.")
             retriever_used = "FAISS (Semantic)"
             try:
                 faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
                 documents = faiss_retriever.get_relevant_documents(query)
                 logger.debug(f"FAISS search returned {len(documents)} documents.")
             except Exception as e:
                  logger.error(f"Error during FAISS fallback search: {e}", exc_info=True)
                  documents = []

        # 3. Try BM25 (if previous attempts failed or returned no results)
        if not documents and self.bm25_retriever:
             logger.info("Previous searches yielded no results. Falling back to BM25 search.")
             retriever_used = "BM25 (Keyword)"
             try:
                 self.bm25_retriever.k = k # Ensure BM25 retriever uses the requested k
                 documents = self.bm25_retriever.get_relevant_documents(query)
                 logger.debug(f"BM25 search returned {len(documents)} documents.")
             except Exception as e:
                  logger.error(f"Error during BM25 fallback search: {e}", exc_info=True)
                  documents = []


        # --- Process and Format Results ---
        if not documents:
             logger.warning(f"No relevant documents found for query '{query}' using any available retriever.")
             return "No relevant context found in the database for your query."

        logger.info(f"Search using '{retriever_used}' returned {len(documents)} candidate documents.")
        # Ensure we only return up to k unique documents (based on content, metadata can differ slightly)
        seen_content = set()
        unique_documents = []
        for doc in documents:
            if doc.page_content not in seen_content:
                unique_documents.append(doc)
                seen_content.add(doc.page_content)
            if len(unique_documents) >= k:
                break

        final_documents = unique_documents
        logger.info(f"Returning top {len(final_documents)} unique documents as context.")

        contexts = []
        for i, doc in enumerate(final_documents):
            metadata = doc.metadata
            context_header = f"--- Context {i+1} (Retriever: {retriever_used}) ---"
            # Build metadata string selectively
            meta_parts = [f"Source: {metadata.get('source_file', 'N/A')}"]
            if metadata.get('match_id'): meta_parts.append(f"MatchID: {metadata['match_id']}")
            if metadata.get('data_type'): meta_parts.append(f"Type: {metadata['data_type']}")
            if metadata.get('match_date'): meta_parts.append(f"Date: {metadata['match_date']}")
            if metadata.get('inning'): meta_parts.append(f"Inning: {metadata['inning']}")
            if metadata.get('teams'): meta_parts.append(f"Teams: {metadata['teams']}")
            if metadata.get('player'): meta_parts.append(f"Player: {metadata['player']}")
            # Add others as needed

            context_body = f"Metadata: [{'; '.join(meta_parts)}]\n"
            context_body += f"Content Chunk:\n{doc.page_content}\n"
            contexts.append(f"{context_header}\n{context_body}")

        return "\n\n".join(contexts)

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the data loader and retrievers."""
        # Safely count files
        try:
             total_files = len(list(self.data_dir.glob('*.json'))) + len(list(self.data_dir.glob('*.jsonl')))
        except Exception:
             total_files = "Error reading data dir"

        # Calculate loading stats based on self.loaded_files_info populated during load_process_and_split_docs
        loaded_ok_count = sum(1 for status in self.loaded_files_info.values() if status == "loaded")
        processed_attempted = len(self.loaded_files_info)
        failed_or_skipped_count = sum(1 for status in self.loaded_files_info.values() if status != "loaded")

        # Try getting embedding dimension safely
        embed_dim = "N/A"
        if self.embeddings:
            try:
                # Prefer specific methods if available
                if hasattr(self.embeddings, 'client') and hasattr(self.embeddings.client, 'get_sentence_embedding_dimension'): # Specific to some HF clients
                    embed_dim = self.embeddings.client.get_sentence_embedding_dimension()
                else:
                    # Generic fallback by embedding a short string
                    dummy_embedding = self.embeddings.embed_query("dimension test")
                    embed_dim = len(dummy_embedding)
            except NotImplementedError:
                 embed_dim = "Not implemented by provider"
            except Exception as e:
                logger.warning(f"Could not determine embedding dimension: {e}")
                embed_dim = "Error Determining"

        status = {
            "config": {
                "data_directory": str(self.data_dir),
                "faiss_persist_directory": str(self.persist_dir),
                "embedding_provider": self.embedding_provider,
                # Use the _effective_model_name determined during init
                "embedding_model": self._effective_model_name or "Not Initialized",
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
            "loading_status": {
                "total_json_files_in_dir": total_files,
                "files_processed_attempted": processed_attempted,
                "files_successfully_loaded": loaded_ok_count,
                "files_failed_or_skipped": failed_or_skipped_count,
                # Counts below are accumulated during _create_documents
                "total_items_processed": self.processed_items_count,
                "total_items_failed": self.failed_items_count,
            },
             # Use new keys matching app.py sidebar structure
             "vector_store_status": {
                 "initialized": self.vector_store is not None,
                 "vector_count": self.vector_store.index.ntotal if self.vector_store else 0,
                 "embedding_dim": embed_dim,
                 "index_on_disk": (self.persist_dir / "index.faiss").exists(),
                 # Add BM25/Ensemble status if needed in sidebar later
                 "bm25_ready": self.bm25_retriever is not None,
                 "ensemble_ready": self.ensemble_retriever is not None,
                 "total_split_chunks": len(self.all_split_texts) # Chunks used for BM25/FAISS build
             },
             # Keep embedding provider/model separate for clarity in sidebar
             "embedding_provider": self.embedding_provider,
             "embedding_model": self._effective_model_name or "Not Initialized"
        }
        return status


# --- Example Usage ---
if __name__ == "__main__":
    # Create dummy data for demonstration
    DEMO_DATA_DIR = Path("./demo_cricket_data_hybrid")
    DEMO_DATA_DIR.mkdir(exist_ok=True)
    DEMO_PERSIST_DIR_HF = Path("./demo_cricket_index_hybrid_hf")
    DEMO_PERSIST_DIR_GOOGLE = Path("./demo_cricket_index_hybrid_google")

    # Sample File 1: Match Summary
    match_summary = {
        "match_id": "AUSvNZ_ODI_2023_01", "data_type": "match_summary", "date": "2023-11-10",
        "teams": ["Australia", "New Zealand"], "venue": "MCG", "result": "Australia won by 50 runs",
        "summary": "David Warner's brilliant century (125 off 110 balls) set Australia up for a commanding total of 320. New Zealand started the chase well with Kane Williamson scoring 85, but Mitchell Starc's 4-wicket haul dismantled their middle order. Australia maintain their dominance in the series."
    }
    with open(DEMO_DATA_DIR / "match_summary_ausvnz.json", 'w') as f: json.dump(match_summary, f, indent=2)

    # Sample File 2: Commentary (JSON Lines)
    commentary_lines = [
        {"match_id": "AUSvNZ_ODI_2023_01", "type": "commentary", "inning": 1, "over": 45, "ball": 2, "bowler": "Trent Boult", "batsman": "David Warner", "text": "FOUR! Warner slashes hard outside off, gets a thick edge that flies past the keeper for a boundary. Moves to 110."},
        {"match_id": "AUSvNZ_ODI_2023_01", "type": "commentary", "inning": 2, "over": 30, "ball": 5, "bowler": "Mitchell Starc", "batsman": "Kane Williamson", "text": "OUT! Bowled 'em! Starc gets the big fish! Williamson plays down the wrong line to a searing yorker, and the off stump is pegged back. Huge moment!"},
        {"match_id": "AUSvNZ_ODI_2023_01", "type": "commentary", "inning": 1, "over": 48, "ball": 1, "bowler": "Tim Southee", "batsman": "Glenn Maxwell", "text": "SIX! What a shot! Maxwell just flicks it over deep square leg. Incredible power!"}
    ]
    with open(DEMO_DATA_DIR / "commentary_ausvnz.jsonl", 'w') as f:
        for line in commentary_lines:
            f.write(json.dumps(line) + '\n')

    # --- Demo Function ---
    def run_demo(provider: Literal['huggingface', 'google'], persist_dir: Path):
        print(f"\n--- Initializing Hybrid DataLoader ({provider.capitalize()}) ---")
        # Clean persist dir for a fresh build in demo
        if persist_dir.exists():
            import shutil
            shutil.rmtree(persist_dir)
            print(f"Cleaned old persist directory: {persist_dir}")

        try:
            # Explicitly set model_name=None to test default behavior
            loader = CricketDataLoaderV2(
                data_dir=str(DEMO_DATA_DIR),
                persist_dir=str(persist_dir),
                embedding_provider=provider,
                model_name=None, # Let it use the default for the provider
                verbose=True
            )

            print("\n--- Loading/Building Vector Store & BM25 Index ---")
            # Set force_rebuild=True to ensure it builds in the demo
            vector_store = loader.load_and_process_data(force_rebuild=True)

            print("\n--- DataLoader Status ---")
            print(json.dumps(loader.get_status(), indent=2, ensure_ascii=False))

            if vector_store: # Check if FAISS was built
                 if loader.ensemble_retriever or loader.vector_store or loader.bm25_retriever:
                     print("\n--- Querying using Hybrid Search (or Fallback) ---")

                     query1 = "Who took 4 wickets for Australia?"
                     context1 = loader.get_relevant_context(query1, k=3, use_hybrid=True)
                     print(f"\nQuery: {query1}\nRetrieved Context:\n{context1}")

                     query2 = "Tell me about David Warner's score"
                     context2 = loader.get_relevant_context(query2, k=3, use_hybrid=True)
                     print(f"\nQuery: {query2}\nRetrieved Context:\n{context2}")

                     query3 = "Exact commentary for Williamson dismissal yorker" # BM25 might help here
                     context3 = loader.get_relevant_context(query3, k=2, use_hybrid=True)
                     print(f"\nQuery: {query3}\nRetrieved Context:\n{context3}")

                     query4 = "Shot over deep square leg" # More semantic / keyword mix
                     context4 = loader.get_relevant_context(query4, k=2, use_hybrid=True)
                     print(f"\nQuery: {query4}\nRetrieved Context:\n{context4}")

                 else:
                     print("\n--- Retrievers not initialized successfully. Cannot query. ---")
            else:
                 print("\n--- FAISS Vector Store failed to initialize. Cannot query. ---")


        except (ImportError, ValueError, RuntimeError, FileNotFoundError) as e:
             print(f"\n--- Demo for '{provider}' failed prerequisite: {e} ---")
             if provider == 'google':
                 print("Ensure 'langchain-google-genai', 'google-generativeai' are installed and GOOGLE_API_KEY is set.")
        except Exception as e:
            print(f"\n--- An unexpected error occurred during the '{provider}' demo: {e} ---")
            logging.error(f"Demo execution failed for {provider}", exc_info=True)


    # --- Run Demos ---
    # Run HuggingFace Demo
    run_demo('huggingface', DEMO_PERSIST_DIR_HF)

    # Run Google Demo (will fail gracefully if key/libs missing)
    print("\n" + "="*50)
    run_demo('google', DEMO_PERSIST_DIR_GOOGLE)
    print("="*50)


    # --- Cleanup (optional) ---
    # uncomment below to delete demo files after run
    # print("\nCleaning up demo directories...")
    # import shutil
    # if DEMO_DATA_DIR.exists(): shutil.rmtree(DEMO_DATA_DIR)
    # if DEMO_PERSIST_DIR_HF.exists(): shutil.rmtree(DEMO_PERSIST_DIR_HF)
    # if DEMO_PERSIST_DIR_GOOGLE.exists(): shutil.rmtree(DEMO_PERSIST_DIR_GOOGLE)
    # print("Cleanup complete.")