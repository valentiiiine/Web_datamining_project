import os
import re
from pathlib import Path
from typing import List, Tuple

import requests
from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef


# ----------------------------
# Configuration
# ----------------------------
TTL_FILE = "kg_artifacts/rag_tennis_kg.ttl"
SOURCE_TTL_FILE = "kg_artifacts/expanded_kg.ttl"
OLLAMA_URL = "http://localhost:11434/api/generate"
GEMMA_MODEL = "gemma:2b"  # If "model not found", try "gemma2:2b" or "llama3.1:8b"

MAX_PREDICATES = 80
MAX_CLASSES = 40
SAMPLE_TRIPLES = 20
OLLAMA_CMD = ["ollama", "run"]  # keep as-is; model name appended later

TENNIS = Namespace("http://example.org/tennis/ontology/")
TEN_ENTITY_PREFIXES = (
    "http://example.org/tennis/player/",
    "http://example.org/tennis/tournament/",
    "http://example.org/tennis/tournamentedition/",
    "http://example.org/tennis/surface/",
    "http://example.org/tennis/country/",
)

PROJECT_ROOT = Path(__file__).resolve().parent


# ----------------------------
# Utility helpers
# ----------------------------

def is_tennis_uri(value) -> bool:
    if not isinstance(value, URIRef):
        return False
    text = str(value)
    return text.startswith(TEN_ENTITY_PREFIXES) or text.startswith(str(TENNIS))


def is_local_entity(value) -> bool:
    if not isinstance(value, URIRef):
        return False
    return str(value).startswith(TEN_ENTITY_PREFIXES)


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_rag_graph(ttl_path: str = TTL_FILE, source_ttl_path: str = SOURCE_TTL_FILE) -> str:
    """
    Create a smaller, tennis-focused RDF graph for the RAG notebook and CLI.
    This keeps the schema stable for a small local model while preserving
    traceable local tennis entities and labels.
    """
    ttl_file = resolve_project_path(ttl_path)
    source_file = resolve_project_path(source_ttl_path)

    if ttl_file.exists():
        return str(ttl_file)

    if not source_file.exists():
        raise FileNotFoundError(
            f"Source graph not found: {source_file}. Run notebook 4 first."
        )

    source_graph = Graph()
    source_graph.parse(source_file, format="turtle")

    rag_graph = Graph()
    rag_graph.bind("tennis", TENNIS)
    rag_graph.bind("rdf", RDF)
    rag_graph.bind("rdfs", RDFS)

    kept_nodes = set()

    for s, p, o in source_graph:
        if is_tennis_uri(p):
            rag_graph.add((s, p, o))
            if isinstance(s, URIRef):
                kept_nodes.add(s)
            if isinstance(o, URIRef):
                kept_nodes.add(o)
            continue

        if p == RDF.type and is_local_entity(s) and is_tennis_uri(o):
            rag_graph.add((s, p, o))
            kept_nodes.add(s)
            kept_nodes.add(o)
            continue

        if p == RDFS.label and is_local_entity(s):
            rag_graph.add((s, p, o))
            kept_nodes.add(s)
            continue

    for node in list(kept_nodes):
        for predicate in (RDFS.label, RDF.type):
            for _, p, o in source_graph.triples((node, predicate, None)):
                if p == RDF.type and not (is_tennis_uri(o) or str(o).startswith("http://www.w3.org/2002/07/owl#")):
                    continue
                rag_graph.add((node, p, o))

    ttl_file.parent.mkdir(parents=True, exist_ok=True)
    rag_graph.serialize(destination=str(ttl_file), format="turtle")
    return str(ttl_file)


def ollama_is_available() -> bool:
    try:
        response = requests.get("http://localhost:11434", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


# ----------------------------
# 0) Utility: call local LLM (Ollama)
# ----------------------------

def ask_local_llm(prompt: str, model: str = GEMMA_MODEL) -> str:
    """
    Send a prompt to a local Ollama model using the REST API.
    Returns the full text response as a single string.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not reach Ollama at http://localhost:11434. "
            "Start the service with `ollama serve` or the desktop app."
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(f"Ollama API error {response.status_code}: {response.text}")

    data = response.json()
    return data.get("response", "")


# ----------------------------
# 1) Load RDF graph
# ----------------------------

def load_graph(ttl_path: str) -> Graph:
    ttl_file = resolve_project_path(ttl_path)
    g = Graph()
    g.parse(ttl_file, format="turtle")
    print(f"Loaded {len(g)} triples from {ttl_file}")
    return g


# ----------------------------
# 2) Build a small schema summary
# ----------------------------

def get_prefix_block(g: Graph) -> str:
    defaults = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "owl": "http://www.w3.org/2002/07/owl#",
        "tennis": str(TENNIS),
    }
    lines = [f"PREFIX {p}: <{ns}>" for p, ns in defaults.items()]
    return "\n".join(sorted(lines))


def list_distinct_predicates(g: Graph, limit=MAX_PREDICATES) -> List[str]:
    q = f"""
    PREFIX tennis: <{TENNIS}>
    SELECT DISTINCT ?p WHERE {{
      ?s ?p ?o .
      FILTER(STRSTARTS(STR(?p), STR(tennis:)))
    }}
    ORDER BY ?p
    LIMIT {limit}
    """
    return [str(row.p) for row in g.query(q)]


def list_distinct_classes(g: Graph, limit=MAX_CLASSES) -> List[str]:
    q = f"""
    PREFIX tennis: <{TENNIS}>
    SELECT DISTINCT ?cls WHERE {{
      ?s a ?cls .
      FILTER(STRSTARTS(STR(?cls), STR(tennis:)))
    }}
    ORDER BY ?cls
    LIMIT {limit}
    """
    return [str(row.cls) for row in g.query(q)]


def sample_triples(g: Graph, limit=SAMPLE_TRIPLES) -> List[Tuple[str, str, str]]:
    q = f"""
    PREFIX tennis: <{TENNIS}>
    SELECT ?s ?p ?o WHERE {{
      ?s ?p ?o .
      FILTER(
        STRSTARTS(STR(?p), STR(tennis:)) ||
        (STRSTARTS(STR(?s), "http://example.org/tennis/") && (?p = rdf:type || ?p = rdfs:label))
      )
    }}
    LIMIT {limit}
    """
    return [(str(r.s), str(r.p), str(r.o)) for r in g.query(q)]


def list_labeled_examples(g: Graph, class_uri: str, limit: int = 10) -> List[str]:
    q = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?label WHERE {{
      ?s a <{class_uri}> ;
         rdfs:label ?label .
    }}
    ORDER BY ?label
    LIMIT {limit}
    """
    return [str(row.label) for row in g.query(q)]


def build_schema_summary(g: Graph) -> str:
    prefixes = get_prefix_block(g)
    preds = list_distinct_predicates(g)
    clss = list_distinct_classes(g)
    samples = sample_triples(g)
    tournaments = list_labeled_examples(g, str(TENNIS.Tournament), limit=8)
    surfaces = list_labeled_examples(g, str(TENNIS.Surface), limit=8)
    players = list_labeled_examples(g, str(TENNIS.Player), limit=12)

    pred_lines = "\n".join(f"- {p}" for p in preds)
    cls_lines = "\n".join(f"- {c}" for c in clss)
    tournament_lines = "\n".join(f"- {item}" for item in tournaments)
    surface_lines = "\n".join(f"- {item}" for item in surfaces)
    player_lines = "\n".join(f"- {item}" for item in players)
    sample_lines = "\n".join(f"- {s} {p} {o}" for s, p, o in samples)

    summary = f"""
{prefixes}

# Domain reminder
- This graph is about tennis Grand Slam tournaments.
- Core classes: Player, Tournament, TournamentEdition, Surface, Country.
- Core predicates: won, participatedIn, playedAgainst, hasSurface, fromCountry, editionYear.
- When matching a player or tournament name, use rdfs:label and lowercase string filters.
- Prefer SELECT DISTINCT queries.
- Prefer simple patterns over complex OPTIONAL chains.

# Predicates (local tennis ontology)
{pred_lines}

# Classes / rdf:type (local tennis ontology)
{cls_lines}

# Tournament label examples
{tournament_lines}

# Surface label examples
{surface_lines}

# Player label examples
{player_lines}

# Sample triples
{sample_lines}
"""
    return summary.strip()


# ----------------------------
# 3) Prompting Gemma: NL → SPARQL
# ----------------------------

SPARQL_INSTRUCTIONS = """
You are a SPARQL generator for a tennis Grand Slam knowledge graph.
Convert the user QUESTION into a valid SPARQL 1.1 SELECT query
for the given RDF graph schema. Follow strictly:

- Use ONLY the IRIs and prefixes visible in the SCHEMA SUMMARY.
- Do NOT invent new predicates or classes.
- Use rdfs:label filters to match names such as Rafael Nadal or Wimbledon Championships.
- For playedAgainst questions, consider both directions with UNION when needed.
- Prefer SELECT DISTINCT.
- Keep the query short, readable, and robust.
- Return ONLY the SPARQL query in a single fenced code block labeled ```sparql
- No explanations or extra text outside the code block.
"""

BASELINE_SYSTEM_CONTEXT = """
Answer from general tennis knowledge only.
The domain is tennis, even when the user says "tennis players", "tennis tournaments",
or "tennis surfaces". Do not mention tool limitations or internet access.
If you are unsure, give your best concise tennis answer.
""".strip()


def make_sparql_prompt(schema_summary: str, question: str) -> str:
    return f"""{SPARQL_INSTRUCTIONS}

SCHEMA SUMMARY:
{schema_summary}

QUESTION:
{question}

Return only the SPARQL query in a code block.
"""


CODE_BLOCK_RE = re.compile(r"```(?:sparql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sparql_from_text(text: str) -> str:
    m = CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def normalize_name_for_filter(text: str) -> str:
    return text.strip().lower()


def normalize_question_for_patterns(question: str) -> str:
    normalized = question.strip().rstrip("?")
    normalized = re.sub(r"\btennis\s+(players?|tournaments?)\b", r"\1", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\btennis\s+surface\b", "surface", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def looks_like_sparql(query: str) -> bool:
    stripped = query.strip()
    if not stripped:
        return False
    first_line = stripped.splitlines()[0].strip().upper()
    return first_line.startswith("PREFIX") or first_line.startswith("SELECT")


def fallback_sparql_from_question(question: str) -> str | None:
    """
    Deterministic fallback for the most common tennis QA patterns.
    This is used only when the local LLM returns invalid or empty queries.
    """
    q = normalize_question_for_patterns(question)
    q_lower = q.lower()

    match = re.search(r"which tournaments did (.+) win$", q_lower)
    if match:
        player_name = normalize_name_for_filter(match.group(1))
        return f"""
PREFIX tennis: <{TENNIS}>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?tournamentLabel WHERE {{
  ?player a tennis:Player ;
          rdfs:label ?playerLabel ;
          tennis:won ?tournament .
  ?tournament rdfs:label ?tournamentLabel .
  FILTER(LCASE(STR(?playerLabel)) = "{player_name}")
}}
ORDER BY ?tournamentLabel
""".strip()

    match = re.search(r"which players are from (.+)$", q_lower)
    if match:
        country_name = normalize_name_for_filter(match.group(1))
        return f"""
PREFIX tennis: <{TENNIS}>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?playerLabel WHERE {{
  ?player a tennis:Player ;
          rdfs:label ?playerLabel ;
          tennis:fromCountry ?country .
  ?country rdfs:label ?countryLabel .
  FILTER(LCASE(STR(?countryLabel)) = "{country_name}")
}}
ORDER BY ?playerLabel
""".strip()

    match = re.search(r"which players participated in (?:the )?(.+)$", q_lower)
    if match:
        tournament_name = normalize_name_for_filter(match.group(1))
        return f"""
PREFIX tennis: <{TENNIS}>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?playerLabel WHERE {{
  ?player a tennis:Player ;
          rdfs:label ?playerLabel ;
          tennis:participatedIn ?tournament .
  ?tournament rdfs:label ?tournamentLabel .
  FILTER(CONTAINS(LCASE(STR(?tournamentLabel)), "{tournament_name}"))
}}
ORDER BY ?playerLabel
LIMIT 20
""".strip()

    match = re.search(r"which players played against (.+)$", q_lower)
    if match:
        player_name = normalize_name_for_filter(match.group(1))
        return f"""
PREFIX tennis: <{TENNIS}>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?opponentLabel WHERE {{
  {{
    ?player rdfs:label ?playerLabel ;
            tennis:playedAgainst ?opponent .
    FILTER(LCASE(STR(?playerLabel)) = "{player_name}")
  }}
  UNION
  {{
    ?opponent tennis:playedAgainst ?player .
    ?player rdfs:label ?playerLabel .
    FILTER(LCASE(STR(?playerLabel)) = "{player_name}")
  }}
  ?opponent rdfs:label ?opponentLabel .
}}
ORDER BY ?opponentLabel
LIMIT 20
""".strip()

    match = re.search(r"which surface is (?:the )?(.+?) played on$", q_lower)
    if match:
        tournament_name = normalize_name_for_filter(match.group(1))
        return f"""
PREFIX tennis: <{TENNIS}>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?surfaceLabel WHERE {{
  ?tournament a tennis:Tournament ;
              rdfs:label ?tournamentLabel ;
              tennis:hasSurface ?surface .
  ?surface rdfs:label ?surfaceLabel .
  FILTER(CONTAINS(LCASE(STR(?tournamentLabel)), "{tournament_name}"))
}}
ORDER BY ?surfaceLabel
""".strip()

    return None


def generate_sparql(question: str, schema_summary: str, model: str = GEMMA_MODEL) -> str:
    raw = ask_local_llm(make_sparql_prompt(schema_summary, question), model=model)
    query = extract_sparql_from_text(raw)
    if not looks_like_sparql(query) or "select" not in query.lower():
        fallback_query = fallback_sparql_from_question(question)
        if fallback_query:
            return fallback_query
    return query


# ----------------------------
# 4) Execute SPARQL with rdflib (and optional self-repair)
# ----------------------------

def run_sparql(g: Graph, query: str) -> Tuple[List[str], List[Tuple]]:
    res = g.query(query)
    vars_ = [str(v) for v in res.vars]
    rows = [tuple(str(cell) for cell in r) for r in res]
    return vars_, rows


REPAIR_INSTRUCTIONS = """
The previous SPARQL failed to execute. Using the SCHEMA SUMMARY and the ERROR MESSAGE,
return a corrected SPARQL 1.1 SELECT query. Follow strictly:

- Use only known prefixes and predicates from the schema summary.
- Keep it simple and robust.
- If name matching is needed, use rdfs:label and lowercase filters.
- Return only a single code block with the corrected SPARQL.
"""


def repair_sparql(
    schema_summary: str,
    question: str,
    bad_query: str,
    error_msg: str,
    model: str = GEMMA_MODEL,
) -> str:
    prompt = f"""{REPAIR_INSTRUCTIONS}

SCHEMA SUMMARY:
{schema_summary}

ORIGINAL QUESTION:
{question}

BAD SPARQL:
{bad_query}

ERROR MESSAGE:
{error_msg}

Return only the corrected SPARQL in a code block.
"""
    raw = ask_local_llm(prompt, model=model)
    repaired = extract_sparql_from_text(raw)
    if not looks_like_sparql(repaired) or "select" not in repaired.lower():
        fallback_query = fallback_sparql_from_question(question)
        if fallback_query:
            return fallback_query
    return repaired


# ----------------------------
# 5) Orchestration: Ask with SPARQL-generation RAG
# ----------------------------

def answer_with_sparql_generation(
    g: Graph,
    schema_summary: str,
    question: str,
    try_repair: bool = True,
    model: str = GEMMA_MODEL,
) -> dict:
    sparql = generate_sparql(question, schema_summary, model=model)

    try:
        vars_, rows = run_sparql(g, sparql)
        if rows:
            return {"query": sparql, "vars": vars_, "rows": rows, "repaired": False, "error": None}
        fallback_query = fallback_sparql_from_question(question)
        if fallback_query:
            fallback_vars, fallback_rows = run_sparql(g, fallback_query)
            if fallback_rows:
                return {
                    "query": fallback_query,
                    "vars": fallback_vars,
                    "rows": fallback_rows,
                    "repaired": True,
                    "error": None,
                }
        return {"query": sparql, "vars": vars_, "rows": rows, "repaired": False, "error": None}
    except Exception as e:
        err = str(e)
        if try_repair:
            fallback_query = fallback_sparql_from_question(question)
            if fallback_query:
                try:
                    fallback_vars, fallback_rows = run_sparql(g, fallback_query)
                    if fallback_rows:
                        return {
                            "query": fallback_query,
                            "vars": fallback_vars,
                            "rows": fallback_rows,
                            "repaired": True,
                            "error": None,
                        }
                except Exception:
                    pass
            repaired = repair_sparql(schema_summary, question, sparql, err, model=model)
            try:
                vars_, rows = run_sparql(g, repaired)
                if rows:
                    return {"query": repaired, "vars": vars_, "rows": rows, "repaired": True, "error": None}
                fallback_query = fallback_sparql_from_question(question)
                if fallback_query:
                    fallback_vars, fallback_rows = run_sparql(g, fallback_query)
                    if fallback_rows:
                        return {
                            "query": fallback_query,
                            "vars": fallback_vars,
                            "rows": fallback_rows,
                            "repaired": True,
                            "error": None,
                        }
                return {"query": repaired, "vars": vars_, "rows": rows, "repaired": True, "error": None}
            except Exception as e2:
                fallback_query = fallback_sparql_from_question(question)
                if fallback_query:
                    try:
                        fallback_vars, fallback_rows = run_sparql(g, fallback_query)
                        if fallback_rows:
                            return {
                                "query": fallback_query,
                                "vars": fallback_vars,
                                "rows": fallback_rows,
                                "repaired": True,
                                "error": None,
                            }
                    except Exception:
                        pass
                return {"query": repaired, "vars": [], "rows": [], "repaired": True, "error": str(e2)}
        else:
            return {"query": sparql, "vars": [], "rows": [], "repaired": False, "error": err}


# ----------------------------
# 6) (Baseline) Direct LLM answer w/o KG
# ----------------------------

def answer_no_rag(question: str, model: str = GEMMA_MODEL) -> str:
    prompt = (
        f"{BASELINE_SYSTEM_CONTEXT}\n\n"
        "Question:\n"
        f"{question}"
    )
    return ask_local_llm(prompt, model=model)


# ----------------------------
# 7) CLI demo
# ----------------------------

def pretty_print_result(result: dict):
    if result.get("error"):
        print("\n[Execution Error]", result["error"])
    print("\n[SPARQL Query Used]")
    print(result["query"])
    print("\n[Repaired?]", result["repaired"])
    vars_ = result.get("vars", [])
    rows = result.get("rows", [])
    if not rows:
        print("\n[No rows returned]")
        return
    print("\n[Results]")
    print(" | ".join(vars_))
    for r in rows[:20]:
        print(" | ".join(r))
    if len(rows) > 20:
        print(f"... (showing 20 of {len(rows)})")


def main():
    model = os.getenv("OLLAMA_MODEL", GEMMA_MODEL)
    ttl_path = ensure_rag_graph()
    g = load_graph(ttl_path)
    schema = build_schema_summary(g)

    while True:
        q = input("\nQuestion (or 'quit'): ").strip()
        if q.lower() == "quit":
            break

        print("\n--- Baseline (No RAG) ---")
        print(answer_no_rag(q, model=model))

        print("\n--- SPARQL-generation RAG (Local LLM + rdflib) ---")
        result = answer_with_sparql_generation(g, schema, q, try_repair=True, model=model)
        pretty_print_result(result)


if __name__ == "__main__":
    main()
