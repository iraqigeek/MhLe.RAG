# MHLE-RAG: Multiscale, Hierarchical & Layered Embeddings for RAG

## Overview
MHLE-RAG (MuLeRAG) is a prototype designed to parse, analyze, and query codebases across multiple programming languages at various scales. It leverages Tree-sitter for parsing and uses embedding-based search to enable intelligent code querying and augmented generation.

## Key Features
1. **Multiscale Analysis** 🔍: Examines code at repository, file, class, and function levels.
2. **Hierarchical Processing** 🏗️: Recognizes and utilizes the structured nature of code repositories.
3. **Layered Embeddings** 🧩: Creates rich, contextual embeddings that capture code semantics at multiple granularities.
4. **Multi-Language Support** 🌐: Parses and analyzes code in Java, Kotlin, JavaScript, Go, Python, C++, C, and Swift.
5. **Intelligent Querying** 🤖: Allows natural language queries to find relevant code snippets across the codebase.
6. **Augmented Generation** 🚀: Utilizes retrieved context to enhance code generation capabilities.
7. **Dependency Analysis** 🕸️: Generates comprehensive dependency graphs at various scales.
8. **Requirements Integration** 📝: Optionally processes and integrates software requirements for holistic analysis.

## Components
1. **Tree-sitter Integration** 🌳: Uses Tree-sitter grammars for accurate code parsing.
2. **Multiscale AST Traversers** 🛠️: Custom-written for each supported language to extract relevant code information at multiple levels.
3. **Layered Embedding Generation** 📚: Utilizes specified embedding models for hierarchical code representation.
4. **Retrieval Augmented Query Engine** 🔍: Implements similarity search on layered embeddings for efficient and context-aware code retrieval.
5. **Multiscale Graph Generation** 🗺️: Creates JSON representations of code dependencies at various levels of granularity.

## Setup
1. Initialize Tree-sitter grammars:
   ```
   python grammar_utils/language_grammar_builder.py
   ```
2. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```
3. Configure the Ollama backend or adjust the `EMBEDDING_API_URL` and `LLM_API_URL` as needed.

## Usage
1. Process a codebase:
   ```
   python mhle_rag.py process --root_dir /path/to/your/codebase
   ```
2. (Optional) Process requirements:
   ```
   python mhle_rag.py process_requirements --requirements_csv /path/to/requirements.csv
   ```
3. Query the processed codebase:
   ```
   python mhle_rag.py query
   ```

## Key Files
- `mhle_rag.py`: Main script for processing, querying, and generation.
- `grammar_utils/ast_traversers.py`: Contains language-specific multiscale AST traversal logic.
- `assets/`: Directory where processed data (embeddings, multiscale graphs) is stored.

## Customization
- Extend `LANGUAGE_DATA` in the main script to add or modify supported languages.
- Adjust embedding models by modifying `CODE_EMBEDDING_MODEL` and `REQUIREMENT_EMBEDDING_MODEL`.

## Advanced Features
- **Hierarchical Querying** 🏙️: Implements a multi-level approach to code retrieval, considering repo, file, class, and function levels.
- **Dynamic Multiscale Graph Building** 🖼️: Constructs graphs of query results to visualize code relationships across different scales.
- **Context-Aware Extended Retrieval** 🔎: Uses hierarchical dependency information to intelligently broaden the search scope.
- **Augmented Code Generation** 💡: Leverages retrieved context to generate or suggest code improvements.

## Notes
- Ensure sufficient computational resources and disk space for processing and storing multiscale embeddings and hierarchical data.
- The tool's effectiveness scales with the quality of the embedding models and the structure of your codebase.