import os
#import glob
import sys
import re
import csv
import json
import argparse
import pathlib
#import tiktoken
import logging  
import networkx as nx
import numpy as np
#import requests
import asyncio
import httpx
from scipy.spatial.distance import cosine as cos, hamming  as ham
from tree_sitter import Parser, Language

# custom written ast post-processing utils (there are prob better/cleaner ways of doing this?)
# the intent is to make retrieving file properties on demand a brainless & quick task.
# maybe there's a totally big brain way of looking at this in which case pls lmk!
from grammar_utils.TreeNode_module import TreeNode
from grammar_utils.parse_c import *
from grammar_utils.parse_cpp import *
from grammar_utils.parse_go import *
from grammar_utils.parse_java import *
from grammar_utils.parse_js import *
from grammar_utils.parse_kt import *
from grammar_utils.parse_py import *

# Ollama backend-- update accordingly.
EMBEDDING_API_URL = "http://localhost:11434/api/embeddings"
LLM_API_URL = "http://localhost:11434/v1/chat/completions"
CODEBASE_DB_PATH = "rag_assets/codebase_embeddings.db"
FILE_TREE_PATH = "rag_assets/file_trees.json"
FILE_GRAPH_PATH = "rag_assets/full_graph.json"
REPO_README_PATH = "rag_assets/repos_readme.json"
REQUIREMENTS_DB_PATH = "rag_assets/requirements_embeddings.db"
CODE_EMBEDDING_MODEL = "mxbai-embed-large:latest" #768dim
CODE_SUMMERIZATION_MODEL = "qwen2.5-coder:latest"
REQUIREMENT_EMBEDDING_MODEL = "mxbai-embed-large:latest" #768dim
logging.basicConfig(level=logging.DEBUG)

# see: ./grammar_utils/so, what is this .so file?.md
if sys.platform == "win32":
    LANGUAGE_SO_PATH = "./grammar_utils/languages.dll"
else:
    LANGUAGE_SO_PATH = "./grammar_utils/language_grammars.so"
LANGUAGE_DATA = {
    "java": ("java", [".java"]),
    "kotlin": ("kotlin", [".kt"]),
    "javascript": ("javascript", [".js", ".jsx"]),
    "go": ("go", [".go"]),
    "python": ("python", [".py"]),
    "cpp": ("cpp", [".cpp", ".cc", ".cxx"]),
    "c": ("c", [".c"]),
    "markdown": ("markdown", [".md"]),
    #"swift": ("swift", [".swift"])
    }

def has_extension(file_path, extensions):
    for ext in extensions:
        if file_path.endswith(ext):
            return True
    return False

# I've yet to research on the feasibility of differents encoders being a better fit in this project (over gpt-4)
#enc = tiktoken.encoding_for_model("gpt-4")

#def count_tokens(text):
#    return len(enc.encode(text))


def create_language(name):
    return Language(LANGUAGE_SO_PATH, name)

def init_tree_sitter_languages():
    global extension_to_language
    extension_to_language = {
        lang: (create_language(data[0]), data[1])
        for lang, data in LANGUAGE_DATA.items() if lang != 'markdown'
    }

########################################################################################################################
#################################### CRAWKLING AND PARSING ALL THE SUPPORTED CODE ######################################
########################################################################################################################
#                                                                                                                      #
#     1. Crawl and parse the codebases                                                                                 #
#     2. Generate source code ASTs                                                                                     #
#     3. Traverse ASTs, extrect properties and store to custom TreeNode structure.                                     #
#         a. file_trees.json: dictionary containing all the parsed properties from each file.                          #
#     4. Create code-centric, strucurally coherent embeddings                                                          #
#     5. Compute various levels of dependency closure graphs:                                                          #
#         a. repos_graph.json: a repo level view of the relationship between all the repos processed (repo-to-repo).   #                                                                             #
#         b. <REPO_NAME>.json: invididual view of the interdependenies across a single processed repo (file-to-file).  #
#         c. full_graph.json: a world view of all the interdependencies across all the repos processed (file-to-file). #
#         d. repos_readme.json: a repo level view that maps the README docs to each other based on [repos_graph.json]. #
#                                                                                                                      #
########################################################################################################################
#################################### CRAWKLING AND PARSING ALL THE SUPPORTED CODE ######################################
########################################################################################################################

def find_imported_elements(import_stmt, file_trees):
    imported_elements = []
    for file_path, node_tree in file_trees.items():
        if isinstance(node_tree, dict):
            class_names = node_tree.get('class_names', [])
            functions = node_tree.get('functions', [])
            properties = node_tree.get('property_declarations', [])
        else:
            class_names = node_tree.class_names
            functions = node_tree.functions
            properties = node_tree.property_declarations

        # Check if the import statement matches any class, function, or property
        for class_name in class_names:
            if import_stmt in class_name:
                imported_elements.append(f"class:{class_name}|{file_path}")

        for func in functions:
            if import_stmt == func.name:
                imported_elements.append(f"function:{func.name}|{file_path}")

        for prop in properties:
            if import_stmt in prop:
                imported_elements.append(f"property:{prop}|{file_path}")

    return imported_elements


def get_snippet(node_tree, element_type):
    if not node_tree:
        logging.error("Node tree is None or empty.")
        return "Snippet not available"

    logging.debug(f"Node Tree Content: {len(json.dumps(node_tree, indent=2))}")

    parts = element_type.split(":")
    if len(parts) < 2:
        logging.error(f"Element type '{element_type}' does not have a second part.")
        return "Snippet not available"

    element_prefix = parts[0]
    element_name = parts[1]
    logging.debug(f"Processing element type: {element_type}, element name: {element_name}")

    if isinstance(node_tree, dict):
        functions = node_tree.get('functions', [])
        class_names = node_tree.get('class_names', [])
        property_declarations = node_tree.get('property_declarations', [])
        imports = node_tree.get('imports', [])
    else:
        functions = node_tree.functions
        class_names = node_tree.class_names
        property_declarations = node_tree.property_declarations
        imports = node_tree.imports

    if element_prefix == "function":
        for func in functions:
            if func.get('name') == element_name:
                func_body = func.get('body', '')
                return func_body[:200] + "..." if len(func_body) > 200 else func_body
    elif element_prefix == "class":
        if element_name in class_names:
            return f"class {element_name}"
    elif element_prefix == "property":
        for prop in property_declarations:
            if element_name in prop:
                return prop
    elif element_prefix == "import":
        for imp in imports:
            if element_name in imp:
                return imp
            # Check if the element_name is a part of a longer import statement
            if any(part in imp for part in element_name.split('.')):
                return imp

    logging.warning(f"No matching element found for element type '{element_type}'.")
    return "Snippet not available"


def chunk_text(text, tokens_per_chunk=500):
    words = text.split()
    return [' '.join(words[i:i+tokens_per_chunk]) for i in range(0, len(words), tokens_per_chunk)]

def process_code_string(code_string, language, file_path):
    parser = Parser()
    parser.set_language(language)
    tree = parser.parse(bytes(code_string, "utf8"))
    root_node = tree.root_node

    node_tree = TreeNode(file_path=file_path)

    if language.name == "java":
        traverse_tree_java(root_node, bytes(code_string, "utf8"), node_tree, language)
    elif language.name == "kotlin":
        traverse_tree_kt(root_node, bytes(code_string, "utf8"), node_tree, language)
    elif language.name == "javascript":
        traverse_tree_js(root_node, bytes(code_string, "utf8"), node_tree, language)
    elif language.name == "go":
        traverse_tree_go(root_node, bytes(code_string, "utf8"), node_tree, language)
    elif language.name == "python":
        traverse_tree_python(root_node, bytes(code_string, "utf8"), node_tree, language, parser)
    elif language.name == "cpp":
        traverse_tree_cpp(root_node, bytes(code_string, "utf8"), node_tree, language)
    elif language.name == "c":
        traverse_tree_c(root_node, bytes(code_string, "utf8"), node_tree, language)
    # elif language.name == "swift":
    #     traverse_tree_swift(root_node, bytes(code_string, "utf8"), node_tree, language)
    else:
        raise ValueError(f"Unsupported language: {language.name}")

    return node_tree

def load_file_trees(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    return {}

def should_skip_path(path):
    skip_directories = [
        '.venv','node_modules', 'build', 'dist', 'out', 'bin', '.git', '.svn', '.vscode',
        '__pycache__', '.idea', 'obj', 'lib', 'vendor', 'target', '.next', 'pkg',
        'venv', '.tox', 'wheels', 'Debug', 'Release', 'deps', 'rag_assets'
    ]
    skip = any(skip_dir in path.split(os.path.sep) for skip_dir in skip_directories)
    return skip
def save_file_trees(root_dir, file_trees):
    file_path = os.path.join(os.path.abspath(root_dir), FILE_TREE_PATH)
    path = os.path.dirname(file_path)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(file_path), "w+") as file:
        json.dump({k: v.to_dict() for k, v in file_trees.items()}, file, indent=4)

def extract_component_name(file_path):
    match = re.search(r"/([^/]+)/(?:app/)?src/", file_path)
    if match:
        return match.group(1)
    return None

def save_file_graph(root_dir, graph_data):
    with open(os.path.join(root_dir, FILE_GRAPH_PATH), "w+") as outfile:
        json.dump(graph_data, outfile, indent=4, sort_keys=True)

def save_repo_readme(root_dir, readme_info_list):
    with open(os.path.join(root_dir, REPO_README_PATH), 'w', encoding='utf-8') as file:
        json.dump(readme_info_list, file, ensure_ascii=False, indent=4)

async def process_codebase(root_dir):
    init_tree_sitter_languages()
    
    modules = {}
    file_trees = {}
    file_sizes = {}
    package_names = {}
    #file_extensions = [x for xs in [value for key, value in list(LANGUAGE_DATA.values())] for x in xs]
    all_directories = os.listdir(root_dir)
    all_directories.append(root_dir)
    directories = [os.path.join(root_dir, d) 
                   for d in all_directories
                   if (os.path.isdir(os.path.join(root_dir, d)) 
                        and not should_skip_path(os.path.join(root_dir, d)) 
                       #or (os.path.isfile(d) and has_extension(d, file_extensions))
                       )]
    total_directories = len(directories)
    processed_directories = 0
    readme_info_list = []
    global embeddings_db
    embeddings_db = {}

    tasks = []
    for dir_name in root_dir:
        repo_path = os.path.join(root_dir, dir_name)
        if os.path.isdir(repo_path):
            module_dir = dir_name
            processed_directories += 1
            logging.info(f"Processing {dir_name}: {(processed_directories / total_directories) * 100:.2f}% complete")
            task = asyncio.create_task(process_repository(repo_path, modules, file_trees, file_sizes, package_names, readme_info_list))
            tasks.append(task)


    #await asyncio.sleep(10)
    await asyncio.gather(*tasks)
    save_file_trees(root_dir, file_trees)
    save_embeddings_db(root_dir, embeddings_db)

    json_data = process_full_graph(file_trees)

    
    save_file_graph(root_dir, json_data)

    save_repo_readme(root_dir, readme_info_list)    

    generate_individual_user_jsons(json_data)
    generate_root_level_json(json_data)

    #return modules, file_sizes, package_names, file_trees, json_data
    
    print("Codebase processing complete. Embeddings have been saved.")

async def generate_embeddings(text, embedding_model=CODE_EMBEDDING_MODEL):
    headers = {'Content-Type': 'application/json'}
    embeddings = None
    try:
        async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
            embedding_payload = json.dumps({"model": embedding_model, "prompt": text})
            response = await client.post(EMBEDDING_API_URL, data=embedding_payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            embeddings = result.get('embedding')
            if not embeddings:
                logging.error(f"Invalid embedding format received. Expected 768 dimensions, got {len(embeddings) if embeddings else 'None'}")
            else:
                embeddings = np.array(embeddings, dtype=np.float32)
    except Exception as e:
        logging.error(f"Error in generate_embeddings: {str(e)}")
        return None
    return embeddings

async def generate_summaries(func, node_tree, file_path, summary_model=CODE_SUMMERIZATION_MODEL):
    funcs = node_tree.functions
    key = f"function:{func.name}|class:{func.class_name}|path:{file_path}"
    text = f"{key}: {func.body}"
    headers = {'Content-Type': 'application/json'}
    summary = None
    try:
        async with httpx.AsyncClient(verify=False, timeout=300.0) as client:
            prompt = """Act as an expert software developer.
generate a summary text of the following function that describe what the function does, what parameters it takes as input, 
what is the meaning/role/function of each parameter, 
and what it generates or returns also describing the meaning/role/function of each return parameter, 
along with any data types if the datatype is clear or can be inferred without doubt.
Your answer should contain only the summary text without any additional information.
"""
            data = {
                'model': summary_model,
                'stream': False,
                'messages': [
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': text}
                ]
            }
            response = await client.post(LLM_API_URL, data=json.dumps(data), headers=headers)
            response.raise_for_status()
            result = response.json()
            summary = result.get('choices')[0]['message']['content']
            if not summary:
                logging.error(f"Empty summary received!")
    except Exception as e:
        logging.error(f"Error in generate_embeddings: {str(e)}")
        return None
    return summary

def query_embeddings(root_dir, query_text, code_embeddings_db, requirements_db, file_trees, top_k=5):
    file_trees = load_file_trees(os.path.join(root_dir, FILE_TREE_PATH))
    query_embedding = generate_embeddings(query_text)
    if query_embedding is None:
        return [], []

    code_results = []
    requirement_results = []

    # Query code embeddings
    for key, embedding in code_embeddings_db.items():
        if embedding is not None:
            similarity = 1 - cos(query_embedding, embedding)
            file_path = key.split('|path:')[-1]
            snippet = get_snippet(file_trees.get(file_path), key.split('|')[0])
            code_results.append((key, similarity, snippet, "code"))

    # Query requirements embeddings
    for requirement_id, data in requirements_db.items():
        embedding = np.array(data.get("embedding", []))
        if embedding.size == 0:
            continue
        similarity = 1 - cos(query_embedding, embedding)
        requirement_results.append((requirement_id, similarity, data, "requirement"))

    code_results.sort(key=lambda x: x[1], reverse=True)
    requirement_results.sort(key=lambda x: x[1], reverse=True)

    return code_results[:top_k], requirement_results[:top_k]



def layered_query_embeddings(query_text, embeddings_db, file_trees, top_k=5, min_repos=2, merge_mode='overall'):
    query_embedding = generate_embeddings(query_text)
    if query_embedding is None:
        return {}

    all_results = []
    for key, embedding in embeddings_db.items():
        if embedding is not None:
            similarity = 1 - cos(query_embedding, embedding)
            file_path = key.split('|path:')[-1]
            repo_name = file_path.split(os.sep)[0]
            snippet = get_snippet(file_trees.get(file_path), key.split('|')[0])
            all_results.append((key, similarity, snippet, repo_name))

    all_results.sort(key=lambda x: x[1], reverse=True)

    top_results = []
    unique_repos = set()
    for result in all_results:
        top_results.append(result)
        unique_repos.add(result[3])
        if len(top_results) >= top_k and len(unique_repos) >= min_repos:
            break

    final_results = top_results[:top_k]

    return organize_results(file_trees, final_results, top_k)








async def process_repository(repo_path, modules, file_trees, file_sizes, package_names, readme_info_list):
    for root, dirs, files in os.walk(repo_path):
        if should_skip_path(root):
            continue
        logging.debug(f"root: {root}")
        logging.debug(f"dirs: {dirs}")
        logging.debug(f"files: {files}")

        for file in files:
            file_path = os.path.join(root, file)
            logging.debug(f"processing file: {file_path}")
            await process_file(file_path, modules, file_trees, file_sizes, package_names, readme_info_list)


async def process_file(file_path, modules, file_trees, file_sizes, package_names, readme_info_list):
    _, file_extension = os.path.splitext(file_path)

    for lang, (language_obj, extensions) in extension_to_language.items():
        processed = False
        if file_extension in extensions:
            try:
                processed = True
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()

                node_tree = process_code_string(file_content, language_obj, file_path)
                file_trees[file_path] = node_tree
                package_names[file_path] = "/".join(os.path.relpath(file_path, start=os.path.dirname(file_path)).split(os.sep)[:-1])
                file_sizes[file_path] = len(file_content.encode("utf-8")).__float__()

                await manage_embeddings(node_tree, file_path, embeddings_db)

                repo_name = os.path.basename(os.path.dirname(file_path))
                if repo_name not in modules:
                    modules[repo_name] = {}
                modules[repo_name][file_path] = file_content
                
            except UnicodeDecodeError:
                logging.warning(f"Skipping binary file: {file_path}")
                continue

    if "README" in file_path.upper() and not processed:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            readme_info_list.append({"id": file_path, "content": content})


def save_embeddings_db(root_dir, embeddings_db):
    for k, v in embeddings_db.items():
        v.tolist()
    with open(os.path.join(os.path.abspath(root_dir), CODEBASE_DB_PATH), "w+") as file:
        json.dump({k: v.tolist() for k, v in embeddings_db.items()}, file)

def load_embeddings_db(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return {k: np.array(v) for k, v in json.load(file).items()}
    return {}

def build_dynamic_graph(query_results, file_trees):
    G = nx.DiGraph()

    for key, similarity, node_tree in query_results:
        # Extract the code element type and name from the key
        element_type, element_name, file_path = parse_key(key)
        node_id = f"{element_type}:{element_name}|{file_path}"
        G.add_node(node_id, similarity=similarity, type=element_type, name=element_name, file=file_path)

        if isinstance(node_tree, dict):
            imports = node_tree.get('imports', [])
        elif hasattr(node_tree, 'imports'):
            imports = node_tree.imports
        else:
            imports = []

        for import_stmt in imports:
            imported_elements = find_imported_elements(import_stmt, file_trees)
            for imported_element in imported_elements:
                G.add_edge(node_id, imported_element)

    return G if G.nodes else None

def print_graph(G):
    for node in G.nodes:
        node_data = G.nodes[node]
        print(f"Node: {node_data['type']}:{node_data['name']}")
        print(f"  File: {node_data['file']}")
        print(f"  Similarity: {node_data['similarity']:.4f}")
        print("  Edges:")
        for neighbor in G.neighbors(node):
            neighbor_data = G.nodes[neighbor]
            print(f"    -> {neighbor_data['type']}:{neighbor_data['name']} in {neighbor_data['file']}")
        print()

########################################################################################################################
#################################### CODE FOR QUERYING YOUR EMBEDDINGS DATABASE ########################################
########################################################################################################################
#                                                                                                                      #
# If you are running this for the first time you will need to index your repos first. Run the following:               #
#   > python3 multiscale_tree.py process --root_dir /path/to/your/folder/with/projects                                        #
#                                                                                                                      #
# once your `codebase_embeddings.db` is generated you may start querying your embeddings to retrieve top_k code refs   #
#                                                                                                                      #
#                                          >>>>>> EXAMPLE QUERY RUN <<<<<<                                             #
# ❯ python3 multiscale_tree.py query                                                      #
# Embeddings loaded. Ready for queries.                                                                                #
# Enter your queries (type 'exit' to quit):                                                                            #
# Query: how is the ambient light calibration done?                                                                    #
# DEBUG:urllib3.connectionpool:Starting new HTTP connection (1): localhost:11434                                       #
# DEBUG:urllib3.connectionpool:http://localhost:11434 "POST /api/embeddings HTTP/11" 200 None                          #
#                                                                                                                      #
# Top 5 results for query 'how is the ambient light calibration done?':                                                #
# Similarity: 0.6427 - function:ledCalibration|class:|path:/Users/.../lightSensor/sweepLEDBrightness.c.                #
# Similarity: 0.5968 - function:updateAmbienceLight|class:|path:/Users/.../lightSensor/lightSensor.c                   #
# Similarity: 0.5939 - function:ambientLight_sensor_initialize|class:|path:/Users/.../src/lightSensor/lightSensor.c.   #
# Similarity: 0.5765 - property:extern bool isCalibrationEnabled;|path:/Users/.../lightSensor/sweepLEDBrightness.c     #
# Similarity: 0.5720 - property:uint16_t ambienceValue = 0;|path:/Users/.../lightSensor/sweepLEDBrightness.c           #
#                                                                                                                      #
# Query: exit                                                                                                           #
# Querying done!                                                                                                       #
#                                                                                                                      #
########################################################################################################################
#################################### CODE FOR QUERYING YOUR EMBEDDINGS DATABASE ########################################
########################################################################################################################
def parse_key(key):
    parts = key.split('|')
    element_info = parts[0].split(':')
    element_type = element_info[0]
    element_name = ':'.join(element_info[1:])
    file_path = parts[1].split('path:')[1]
    return element_type, element_name, file_path


def organize_results(file_trees, all_results, top_k):
    top_repos = {}
    overall_top_elements = []
    repo_specific_elements = {}
    file_specific_elements = {}

    for key, similarity, node_tree, repo_name in all_results:
        element_type, element_name, file_path = parse_key(key)

        # Create element info dictionaryclass_names
        element_info = {
            "type": element_type,
            "name": element_name,
            "similarity": similarity,
            "file_path": key.split('|path:')[-1],
            "snippet": get_snippet(file_trees.get(file_path), key.split('|')[0]),
            "repo_name": repo_name
        }

        # Add to top repos
        if repo_name not in top_repos:
            top_repos[repo_name] = {"similarity": similarity, "top_files": {}}

        # Add to top files within repo
        if file_path not in top_repos[repo_name]["top_files"]:
            top_repos[repo_name]["top_files"][file_path] = {"similarity": similarity, "top_elements": []}

        # Add to top elements within file
        if len(top_repos[repo_name]["top_files"][file_path]["top_elements"]) < top_k:
            top_repos[repo_name]["top_files"][file_path]["top_elements"].append(element_info)

        # Add to overall top elements
        if len(overall_top_elements) < top_k:
            overall_top_elements.append(element_info)

        # Add to repo-specific elements
        if repo_name not in repo_specific_elements:
            repo_specific_elements[repo_name] = []
        if len(repo_specific_elements[repo_name]) < top_k:
            repo_specific_elements[repo_name].append(element_info)

        # Add to file-specific elements
        if file_path not in file_specific_elements:
            file_specific_elements[file_path] = []
        if len(file_specific_elements[file_path]) < top_k:
            file_specific_elements[file_path].append(element_info)

    return {
        "top_repos": format_top_repos(top_repos, top_k),
        "overall_top_elements": overall_top_elements,
        "repo_specific_elements": repo_specific_elements,
        "file_specific_elements": file_specific_elements
    }
def format_top_repos(top_repos, top_k):
    formatted_repos = [
        {
            "repo_name": repo,
            "similarity": data["similarity"],
            "top_files": [
                {
                    "file_path": file,
                    "similarity": file_data["similarity"],
                    "top_elements": file_data["top_elements"]
                }
                for file, file_data in data["top_files"].items()
            ]
        }
        for repo, data in top_repos.items()
    ]
    return sorted(formatted_repos, key=lambda x: x["similarity"], reverse=True)[:top_k]

embeddings_db = {}

def process_full_graph(file_tree, file_paths=None):
    
    if file_paths:
        parsed_data = {k: v for k, v in file_tree.items() if k in file_paths}
        #logging.info(f"Filtered parsed_data: {parsed_data}")
    else:
        parsed_data = {k: v for k, v in file_tree.items()}

    links = set()
    dependencies = {}

    for file_path, node_tree in parsed_data.items():
        if not isinstance(node_tree, dict):
            continue

        node_imports = extract_imports(node_tree)
        property_dependencies = extract_property_dependencies(node_tree)
        file_dependencies = []

        current_file_base = os.path.splitext(os.path.basename(file_path))[0]
        logging.info(f"Processing file: {file_path}")

        for other_file_path, other_node_tree in parsed_data.items():
            if file_path == other_file_path or not isinstance(other_node_tree, dict):
                continue

            other_file_base = os.path.splitext(os.path.basename(other_file_path))[0]

            if any(imp.endswith(other_file_base) for imp in node_imports) or \
               any(prop == other_file_base for prop in property_dependencies) or \
               any(other_file_base in func.get('name', '') for func in node_tree.get('functions', [])) or \
               any(other_file_base in class_name for class_name in node_tree.get('class_names', [])):
                file_dependencies.append(other_file_path)
                links.add((file_path, other_file_path))

        dependencies[file_path] = list(set(file_dependencies))
        logging.info(f"Dependencies for {file_path}: {file_dependencies}")

    nodes = []
    for file_path in parsed_data.keys():
        node = {
            "id": file_path,
            "user": extract_component_name(file_path),
            "description": "",
            "fileSize": os.path.getsize(file_path),
        }
        nodes.append(node)
        logging.info(f"Added node: {node}")

    unique_links = [{"source": source, "target": target} for source, target in links]
    logging.info(f"Unique links: {unique_links}")
    return {"nodes": nodes, "links": unique_links}

def extract_imports(node_tree):
    imports = []
    for imp in node_tree.get("imports", []):
        if imp.startswith("import "):
            module = imp.replace("import ", "").strip().split()[0]
            imports.append(module)
        else:
            imports.append(imp.strip())
    return imports

def extract_property_dependencies(node_tree):
    properties = []
    for prop in node_tree.get("property_declarations", []):
        if "@ObservedObject" in prop or "@State" in prop or "@EnvironmentObject" in prop or "@Binding" in prop:
            property_name = re.findall(r'\b\w+\b', prop)[-1]
            properties.append(property_name)
    return properties

# def manage_embeddings(tree_node, file_path, embeddings_db):
#     for class_name in tree_node.class_names:
#         key = f"class:{class_name}|path:{file_path}"
#         embeddings_db[key] = generate_embeddings(f"{key}: {class_name}")

#     for import_stmt in tree_node.imports:
#         key = f"import:{import_stmt}|path:{file_path}"
#         embeddings_db[key] = generate_embeddings(f"{key}: {import_stmt}")

#     for export_stmt in tree_node.exports:
#         key = f"export:{export_stmt}|path:{file_path}"
#         embeddings_db[key] = generate_embeddings(f"{key}: {export_stmt}")

#     for prop in tree_node.property_declarations:
#         key = f"property:{prop}|path:{file_path}"
#         embeddings_db[key] = generate_embeddings(f"{key}: {prop}")

#     for func in tree_node.functions:
#         key = f"function:{func.name}|class:{func.class_name}|path:{file_path}"
#         embeddings_db[key] = generate_embeddings(f"{key}: {func.name}")

#         body_chunks = chunk_text(func.body)
#         for i, chunk in enumerate(body_chunks):
#             key = f"function_{func.name}_body_chunk_{i}|class:{func.class_name}|path:{file_path}"
#             embeddings_db[key] = generate_embeddings(f"{key}: {chunk}")


async def manage_embeddings(node_tree, file_path, embeddings_db):
    # Create a list to hold all tasks
    tasks = []

    for class_name in node_tree.class_names:
        key = f"class:{class_name}|path:{file_path}"
        task = asyncio.create_task(add_embedding(key, generate_embeddings(f"{key}: {class_name}"), embeddings_db))
        tasks.append(task)

    for import_stmt in node_tree.imports:
        key = f"import:{import_stmt}|path:{file_path}"
        task = asyncio.create_task(add_embedding(key, generate_embeddings(f"{key}: {import_stmt}"), embeddings_db))
        tasks.append(task)

    for export_stmt in node_tree.exports:
        key = f"export:{export_stmt}|path:{file_path}"
        task = asyncio.create_task(add_embedding(key, generate_embeddings(f"{key}: {export_stmt}"), embeddings_db))
        tasks.append(task)

    for prop in node_tree.property_declarations:
        key = f"property:{prop}|path:{file_path}"
        task = asyncio.create_task(add_embedding(key, generate_embeddings(f"{key}: {prop}"), embeddings_db))
        tasks.append(task)

    for func in node_tree.functions:
        key = f"function:{func.name}|class:{func.class_name}|path:{file_path}"
        task = asyncio.create_task(add_embedding(key, generate_embeddings(f"{key}: {func.name}"), embeddings_db))
        tasks.append(task)
    # for func in node_tree.functions:
    #     task = asyncio.create_task(add_summary(key, generate_summaries(func, node_tree, file_path), func))
    #     tasks.append(task)

        #body_chunks = chunk_text(func.body)
        #for i, chunk in enumerate(body_chunks):
        #    key = f"function_{func.name}_body_chunk_{i}|class:{func.class_name}|path:{file_path}"
        #    task = asyncio.create_task(add_embedding(key, generate_embeddings(f"{key}: {chunk}"), embeddings_db))
        #    tasks.append(task)

    # Await all the tasks to complete and add their results to the database
    await asyncio.gather(*tasks)

async def add_embedding(key, func, embeddings_db):
    embeddings_db[key] = await func

async def add_summary(key, func, tree_func):
    tree_func.summary = await func

def extended_retrieval(file_tree, top_k, root_dir):
    # Compute dependency graph for initial_files
    initial_files = sorted(file_tree.keys())
    dependency_graph = process_full_graph(file_tree, initial_files)
    dependencies = {node['id'] for node in dependency_graph['nodes']}

    # Add dependencies to initial_files
    extended_files = set(initial_files).union(dependencies)

    # Perform retrieval to get twice the top_k from extended files
    sorted_extended_files = sorted(extended_files)
    return sorted_extended_files[:top_k * 2]

def generate_individual_user_jsons(json_data):
    nodes = json_data['nodes']
    links = json_data['links']

    link_counts = calculate_link_counts(nodes, links)
    for node in nodes:
        node['linkCount'] = link_counts[node['id']]

    user_nodes_dict = {}
    for node in nodes:
        user = node['user']
        if user not in user_nodes_dict:
            user_nodes_dict[user] = []
        user_nodes_dict[user].append(node)

    script_location = pathlib.Path(__file__).parent.absolute()
    assets_dir = script_location / 'rag_assets/files'
    assets_dir.mkdir(parents=True, exist_ok=True)

    file_json = {}
    for user, user_nodes in user_nodes_dict.items():
        user_links = [
            link for link in links
            if link['source'] in [node['id'] for node in user_nodes]
            or link['target'] in [node['id'] for node in user_nodes]
        ]
        file_json = {
            'nodes': user_nodes,
            'links': user_links
        }
        file_path = assets_dir / f'{user}.json'
        with open(file_path, 'w') as outfile:
            json.dump(file_json, outfile, indent=4, sort_keys=True)

    return file_json

def generate_root_level_json(json_data):
    users = {node['user'] for node in json_data['nodes']}
    new_nodes = [
        {
            'id': user,
            'description': user,
            'fileSize': sum(node['fileSize'] for node in json_data['nodes'] if node['user'] == user),
            'fileCount': sum(1 for node in json_data['nodes'] if node['user'] == user)
        }
        for user in users
    ]

    links = set()
    for link in json_data['links']:
        source_user = next(node['user'] for node in json_data['nodes'] if node['id'] == link['source'])
        target_user = next(node['user'] for node in json_data['nodes'] if node['id'] == link['target'])
        if source_user != target_user:
            links.add((source_user, target_user))

    new_links = [{'source': link[0], 'target': link[1]} for link in links]
    repo_json = {
        'nodes': new_nodes,
        'links': new_links
    }

    script_location = pathlib.Path(__file__).parent.absolute()
    assets_dir = script_location / 'rag_assets'
    assets_dir.mkdir(parents=True, exist_ok=True)

    file_path = assets_dir / 'repos_graph.json'
    with open(file_path, 'w') as outfile:
        json.dump(repo_json, outfile, indent=4, sort_keys=True)

    return repo_json

def calculate_link_counts(nodes, links):
    link_counts = {node['id']: 0 for node in nodes}
    for link in links:
        if link['source'] in link_counts:
            link_counts[link['source']] += 1
        if link['target'] in link_counts:
            link_counts[link['target']] += 1
    return link_counts

def interactive_query_mode(root_dir):
    file_trees = load_file_trees(os.path.join(root_dir, FILE_TREE_PATH))
    embeddings_db = load_embeddings_db(os.path.join(root_dir,CODEBASE_DB_PATH))
    requirements_db = load_requirements_db(os.path.join(root_dir, REQUIREMENTS_DB_PATH))

    print("Embeddings loaded. Ready for queries.")
    print("Enter your queries (type 'exit' to quit):")

    while True:
        query = input("Query: ").strip()
        if query.lower() == 'exit':
            break

        top_k = int(input("Enter the number of top results to retrieve: "))

        # Step 1: Extended Retrieval using the dependency graph
        print("\nExtended Retrieval Phase")
        extended_files = extended_retrieval(file_trees, top_k, root_dir)
        logging.info(f"Extended files retrieved: {extended_files}")

        # Step 2: Query using the extended files
        print("\nQuerying Phase")
        logging.info("Starting querying phase")
        code_results, req_results = query_embeddings(query, embeddings_db, requirements_db, {k: file_trees[k] for k in extended_files}, top_k)
        logging.info(f"Results retrieved: {code_results}")

        print(f"\nTop {top_k} code results for query '{query}':")
        for key, similarity, snippet, result_type in code_results:
            print(f"Similarity: {float(similarity):.4f} - {key}")
            print(f"Snippet: {snippet[:100]}...")  # Display first 100 characters of the snippet
            print(f"Type: {result_type}")
            print()

        print(f"\nTop {top_k} requirement results for query '{query}':")
        for req_id, similarity, data, result_type in req_results:
            print(f"Similarity: {float(similarity):.4f} - Requirement ID: {req_id}")
            print(f"Description: {data['data']['Description'][:100]}...")  # Display first 100 characters of the description
            print(f"Type: {result_type}")
            print()

        while True:
            expand = input("\nWould you like to expand the results? (yes/no): ").strip().lower()
            if expand == 'no':
                break
            elif expand == 'yes':
                exclude_file = input("Enter the file path to exclude from results: ").strip()

                if exclude_file in extended_files:
                    extended_files.remove(exclude_file)
                    logging.info(f"Excluding file: {exclude_file}")

                    print("\nExpanded Retrieval Phase")
                    logging.info("Starting expanded retrieval phase")
                    code_results, req_results = query_embeddings(query, embeddings_db, requirements_db, {k: file_trees[k] for k in extended_files}, top_k)
                    logging.info(f"Results retrieved after excluding {exclude_file}: {code_results}")

                    print(f"\nNext {top_k} code results for query '{query}' excluding '{exclude_file}':")
                    for key, similarity, snippet, result_type in code_results:
                        print(f"Similarity: {float(similarity):.4f} - {key}")
                        print(f"Snippet: {snippet[:100]}...")
                        print(f"Type: {result_type}")
                        print()
                else:
                    print(f"File {exclude_file} is not in the current extended file set.")
                    logging.warning(f"Attempted to exclude non-existent file: {exclude_file}")
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")# Function to process requirements from CSV
def process_requirements(csv_file_path):
    requirements_db = {}
    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                requirement_id = row["Project ID"]
                description = row["Description"]
                embedding = generate_embeddings(description, embedding_model=REQUIREMENT_EMBEDDING_MODEL)
                if embedding is not None:
                    requirements_db[requirement_id] = {
                        "embedding": embedding.tolist(),
                        "data": row
                    }
    except Exception as e:
        logging.error(f"Error processing requirements: {str(e)}")

    # Save the requirements embeddings
    save_requirements_db(requirements_db)
    return requirements_db

# Function to save requirements embeddings to disk
def save_requirements_db(requirements_db):
    with open(REQUIREMENTS_DB_PATH, "w") as file:
        json.dump(requirements_db, file)

# Function to load requirements embeddings from disk
def load_requirements_db(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    return {}

def decorator1(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

def decorator2(func):
    def wrapper():
        print("decorator 2.")
        func()
    return wrapper

@decorator1
@decorator2
def say_whee():
    print("Whee!")

@decorator1
def say_whaa():
    print("Whaaa!")

# Main function to process codebase and requirements
def main():
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Code Embedding Processor and Query System")
    parser.add_argument("mode", choices=["process", "query", "process_requirements"], help="Mode of operation: 'process' to analyze codebase, 'query' for interactive querying, 'process_requirements' to process only requirements")
    parser.add_argument("--root_dir", help="Root directory of the codebase (required for 'process' mode)")
    parser.add_argument("--requirements_csv", help="Path to requirements CSV file (required for 'process_requirements' mode, optional for 'process' mode)")

    args = parser.parse_args()

    if args.mode == "process":
        if not args.root_dir:
            print("Error: --root_dir is required for 'process' mode")
            sys.exit(1)
        asyncio.run(process_codebase(args.root_dir))
        if args.requirements_csv:
            process_requirements(args.requirements_csv)
    elif args.mode == "process_requirements":
        if not args.requirements_csv:
            print("Error: --requirements_csv is required for 'process_requirements' mode")
            sys.exit(1)
        process_requirements(args.requirements_csv)
    elif args.mode == "query":
        if not args.root_dir:
            print("Error: --root_dir is required for 'query' mode")
            sys.exit(1)
        file_trees_path = os.path.join(os.path.abspath(args.root_dir), FILE_TREE_PATH)
        if not os.path.exists(file_trees_path):
            print("Error: No file trees found. Please run in 'process' mode first.")
            sys.exit(1)
        interactive_query_mode(args.root_dir)

    say_whee()
if __name__ == "__main__":
    main()
