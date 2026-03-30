from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from difflib import SequenceMatcher

import pandas as pd
import requests
from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, URIRef

from src.kg.kg_construction import EX, ONTO, PROV, PROPERTY_URIS, build_kg_artifacts, role_to_uri


WIKIDATA_ENTITY_PREFIX = "http://www.wikidata.org/entity/"
WD = Namespace(WIKIDATA_ENTITY_PREFIX)
WDT = Namespace("http://www.wikidata.org/prop/direct/")

WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
HEADERS = {
    "User-Agent": "AcademicKGProject/1.0 (knowledge-graph-lab)",
    "Accept": "application/sparql-results+json",
}

ROLE_KEYWORDS = {
    "Player": ["tennis player", "player", "athlete", "sportsman", "sportswoman"],
    "Tournament": ["tennis tournament", "grand slam", "championships", "tournament"],
    "Country": ["country", "sovereign state", "island country", "island state"],
    "Surface": ["court", "surface", "material", "clay", "grass", "hard"],
}

SURFACE_QUERY_OVERRIDES = {
    "Grass": "grass court",
    "Hard Court": "hardcourt",
    "Clay": "clay court",
}

MANUAL_HIGH_CONFIDENCE = {
    ("Rafael Nadal", "Player"): "Q10132",
    ("Novak Djokovic", "Player"): "Q5812",
    ("Carlos Alcaraz", "Player"): "Q85518537",
    ("Roger Federer", "Player"): "Q1426",
    ("Andy Murray", "Player"): "Q33837",
    ("Jannik Sinner", "Player"): "Q64769553",
    ("Alexander Zverev", "Player"): "Q237176",
    ("Daniil Medvedev", "Player"): "Q190239",
    ("Stefanos Tsitsipas", "Player"): "Q310510",
    ("Stan Wawrinka", "Player"): "Q230417",
    ("French Open", "Tournament"): "Q43605",
    ("Australian Open", "Tournament"): "Q60874",
    ("Wimbledon Championships", "Tournament"): "Q41520",
    ("US Open", "Tournament"): "Q123577",
    ("Spain", "Country"): "Q29",
    ("Australia", "Country"): "Q408",
    ("France", "Country"): "Q142",
    ("Serbia", "Country"): "Q403",
}


@dataclass
class AlignmentArtifacts:
    alignment_df: pd.DataFrame
    alignment_graph: Graph
    predicate_alignment_graph: Graph
    expanded_graph: Graph
    expanded_stats: dict


PREDICATE_ALIGNMENT_SPECS = {
    "won": {
        "search_text": "winner",
        "mapping_type": "subPropertyOf",
        "target_property": WDT["P1346"],
        "expected_label": "winner",
        "comment": (
            "Mapped conservatively to Wikidata 'winner' because the local predicate "
            "links a player to a tournament and is more specific than a generic award relation."
        ),
    },
    "participatedIn": {
        "search_text": "participant in",
        "mapping_type": "subPropertyOf",
        "target_property": WDT["P1344"],
        "expected_label": "participant in",
        "comment": (
            "Mapped to Wikidata 'participant in' because the local predicate captures "
            "participation in a tournament event."
        ),
    },
    "fromCountry": {
        "search_text": "country of citizenship",
        "mapping_type": "subPropertyOf",
        "target_property": WDT["P27"],
        "expected_label": "country of citizenship",
        "comment": (
            "Mapped conservatively to 'country of citizenship'. It is not always strictly "
            "identical to nationality in sports text, so subPropertyOf is safer than equivalence."
        ),
    },
    "editionYear": {
        "search_text": "point in time",
        "mapping_type": "subPropertyOf",
        "target_property": WDT["P585"],
        "expected_label": "point in time",
        "comment": (
            "Mapped to 'point in time' because the local editionYear is a simplified temporal "
            "encoding of a tournament edition."
        ),
    },
}


def safe_request_json(url: str, params: dict, timeout: int = 60) -> dict:
    response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.json()


def query_wikidata_candidates(label: str, role: str, limit: int = 5) -> list[dict]:
    search_label = SURFACE_QUERY_OVERRIDES.get(label, label)
    data = safe_request_json(
        WIKIDATA_API_URL,
        {
            "action": "wbsearchentities",
            "language": "en",
            "format": "json",
            "search": search_label,
            "limit": limit,
        },
    )
    return data.get("search", [])


def label_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def role_match_score(description: str | None, role: str) -> float:
    if not description:
        return 0.0
    lowered = description.lower()
    if role == "Country" and any(word in lowered for word in ["city", "municipality", "county", "village", "capital of", "district"]):
        return 0.0
    keywords = ROLE_KEYWORDS.get(role, [])
    if any(keyword in lowered for keyword in keywords):
        return 1.0
    if role == "Player" and "tennis" in lowered:
        return 1.0
    return 0.0


def confidence_score(private_label: str, role: str, candidate: dict) -> float:
    candidate_label = candidate.get("label", "")
    candidate_description = candidate.get("description", "")
    score = 0.65 * label_similarity(private_label, candidate_label)
    score += 0.25 * role_match_score(candidate_description, role)
    if private_label.lower() == candidate_label.lower():
        score += 0.10
    return min(score, 1.0)


def alignment_decision(score: float) -> str:
    if score >= 0.90:
        return "accept_alignment"
    if score >= 0.70:
        return "manual_review"
    return "create_local_entity"


def select_important_entities(cleaned_triples: pd.DataFrame, top_players: int = 40) -> pd.DataFrame:
    rows = []

    player_counts = pd.concat(
        [
            cleaned_triples.loc[cleaned_triples["subject_role"] == "Player", "subject"],
            cleaned_triples.loc[cleaned_triples["object_role"] == "Player", "object"],
        ]
    ).value_counts()
    for label, count in player_counts.head(top_players).items():
        rows.append({"private_entity": label, "entity_role": "Player", "support": int(count)})

    for role in ["Tournament", "Surface", "Country"]:
        role_values = pd.concat(
            [
                cleaned_triples.loc[cleaned_triples["subject_role"] == role, "subject"],
                cleaned_triples.loc[cleaned_triples["object_role"] == role, "object"],
            ]
        ).value_counts()
        for label, count in role_values.items():
            rows.append({"private_entity": label, "entity_role": role, "support": int(count)})

    important_df = pd.DataFrame(rows).drop_duplicates(subset=["private_entity", "entity_role"])
    important_df = important_df.sort_values(["entity_role", "support"], ascending=[True, False]).reset_index(drop=True)
    return important_df


def align_entities(important_df: pd.DataFrame, sleep_seconds: float = 0.05) -> pd.DataFrame:
    alignment_rows = []

    for _, row in important_df.iterrows():
        private_entity = row["private_entity"]
        role = row["entity_role"]

        manual_qid = MANUAL_HIGH_CONFIDENCE.get((private_entity, role))
        if manual_qid:
            alignment_rows.append(
                {
                    "private_entity": private_entity,
                    "entity_role": role,
                    "external_uri": f"{WIKIDATA_ENTITY_PREFIX}{manual_qid}",
                    "candidate_label": private_entity,
                    "candidate_description": "manual high-confidence seed",
                    "confidence": 0.99,
                    "decision": "accept_alignment",
                }
            )
            continue

        candidates = query_wikidata_candidates(private_entity, role)
        if not candidates:
            alignment_rows.append(
                {
                    "private_entity": private_entity,
                    "entity_role": role,
                    "external_uri": None,
                    "candidate_label": None,
                    "candidate_description": None,
                    "confidence": 0.0,
                    "decision": "create_local_entity",
                }
            )
            continue

        scored_candidates = []
        for candidate in candidates:
            score = confidence_score(private_entity, role, candidate)
            scored_candidates.append((score, candidate))

        best_score, best_candidate = max(scored_candidates, key=lambda item: item[0])
        alignment_rows.append(
            {
                "private_entity": private_entity,
                "entity_role": role,
                "external_uri": f"{WIKIDATA_ENTITY_PREFIX}{best_candidate['id']}",
                "candidate_label": best_candidate.get("label"),
                "candidate_description": best_candidate.get("description"),
                "confidence": round(best_score, 3),
                "decision": alignment_decision(best_score),
            }
        )
        time.sleep(sleep_seconds)

    alignment_df = pd.DataFrame(alignment_rows).sort_values(
        ["decision", "entity_role", "confidence"], ascending=[True, True, False]
    )
    return alignment_df.reset_index(drop=True)


def build_alignment_graph(alignment_df: pd.DataFrame) -> Graph:
    g = Graph()
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("ex", EX)
    g.bind("onto", ONTO)
    g.bind("prov", PROV)

    for _, row in alignment_df.iterrows():
        if row["decision"] != "accept_alignment" or not row["external_uri"]:
            continue
        local_uri = role_to_uri(row["entity_role"], row["private_entity"])
        external_uri = URIRef(row["external_uri"])
        g.add((local_uri, OWL.sameAs, external_uri))
        g.add((local_uri, RDFS.label, Literal(row["private_entity"])))
    return g


def run_select_query(query: str, timeout: int = 120) -> list[dict]:
    response = requests.post(
        WIKIDATA_SPARQL_URL,
        data={"query": query, "format": "json"},
        headers=HEADERS,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["results"]["bindings"]


def query_wikidata_property_candidates(search_text: str, limit: int = 5) -> list[dict]:
    # We retrieve candidate properties via SPARQL so the mapping remains auditable
    # and easy to justify in the notebook and final report.
    escaped_text = search_text.replace('"', '\\"').lower()
    query = f"""
    SELECT ?property ?propertyLabel ?directClaim WHERE {{
      ?property a wikibase:Property ;
                wikibase:directClaim ?directClaim ;
                rdfs:label ?propertyLabel .
      FILTER(LANG(?propertyLabel) = "en")
      FILTER(CONTAINS(LCASE(STR(?propertyLabel)), "{escaped_text}"))
    }}
    ORDER BY ?propertyLabel
    LIMIT {limit}
    """
    return run_select_query(query)


def align_predicates() -> Graph:
    graph = Graph()
    graph.bind("owl", OWL)
    graph.bind("rdfs", RDFS)
    graph.bind("onto", ONTO)
    graph.bind("wdt", WDT)

    for local_name, spec in PREDICATE_ALIGNMENT_SPECS.items():
        local_property = PROPERTY_URIS[local_name]
        candidates = query_wikidata_property_candidates(spec["search_text"])

        chosen_candidate = None
        for candidate in candidates:
            label = candidate.get("propertyLabel", {}).get("value", "").lower()
            direct_claim = candidate.get("directClaim", {}).get("value")
            if label == spec["expected_label"] and direct_claim == str(spec["target_property"]):
                chosen_candidate = candidate
                break

        target_property = spec["target_property"]
        predicate = OWL.equivalentProperty if spec["mapping_type"] == "equivalentProperty" else RDFS.subPropertyOf
        graph.add((local_property, predicate, target_property))
        graph.add((local_property, RDFS.comment, Literal(spec["comment"])))

        if chosen_candidate:
            graph.add(
                (
                    local_property,
                    RDFS.seeAlso,
                    URIRef(chosen_candidate["property"]["value"]),
                )
            )

    return graph


def qid_from_uri(uri: str) -> str:
    return uri.rstrip("/").split("/")[-1]


def fetch_one_hop_expansion(seed_uris: list[str], limit: int = 50000) -> list[tuple[str, str, str]]:
    if not seed_uris:
        return []
    values = " ".join(f"wd:{qid_from_uri(uri)}" for uri in seed_uris)
    query = f"""
    SELECT ?s ?p ?o WHERE {{
      VALUES ?s {{ {values} }}
      ?s ?p ?o .
      FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
      FILTER(isIRI(?o))
      FILTER(STRSTARTS(STR(?o), "{WIKIDATA_ENTITY_PREFIX}Q"))
    }}
    LIMIT {limit}
    """
    rows = run_select_query(query)
    return [
        (row["s"]["value"], row["p"]["value"], row["o"]["value"])
        for row in rows
        if row["o"]["value"].startswith(f"{WIKIDATA_ENTITY_PREFIX}Q")
    ]


def fetch_second_hop_expansion(
    object_uris: list[str],
    limit: int = 120000,
    sample_size: int = 1800,
) -> list[tuple[str, str, str]]:
    sampled = object_uris[:sample_size]
    if not sampled:
        return []
    values = " ".join(f"wd:{qid_from_uri(uri)}" for uri in sampled)
    query = f"""
    SELECT ?s ?p ?o WHERE {{
      VALUES ?s {{ {values} }}
      ?s ?p ?o .
      FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
      FILTER(isIRI(?o))
      FILTER(STRSTARTS(STR(?o), "{WIKIDATA_ENTITY_PREFIX}Q"))
    }}
    LIMIT {limit}
    """
    rows = run_select_query(query, timeout=180)
    return [
        (row["s"]["value"], row["p"]["value"], row["o"]["value"])
        for row in rows
        if row["o"]["value"].startswith(f"{WIKIDATA_ENTITY_PREFIX}Q")
    ]


def triples_to_graph(triples: list[tuple[str, str, str]]) -> Graph:
    g = Graph()
    g.bind("wd", WD)
    g.bind("wdt", WDT)
    g.bind("owl", OWL)
    for s, p, o in triples:
        g.add((URIRef(s), URIRef(p), URIRef(o)))
    return g


def filter_external_graph(external_graph: Graph, max_predicates: int = 125) -> Graph:
    predicate_counts: dict[str, int] = {}
    for _, p, _ in external_graph:
        predicate_text = str(p)
        if not predicate_text.startswith("http://www.wikidata.org/prop/direct/"):
            continue
        predicate_counts[predicate_text] = predicate_counts.get(predicate_text, 0) + 1

    kept_predicates = {
        predicate
        for predicate, _ in sorted(
            predicate_counts.items(), key=lambda item: item[1], reverse=True
        )[:max_predicates]
    }

    filtered = Graph()
    filtered.bind("wd", WD)
    filtered.bind("wdt", WDT)
    filtered.bind("owl", OWL)
    for s, p, o in external_graph:
        if str(p) in kept_predicates:
            filtered.add((s, p, o))
    return filtered


def merge_graphs(*graphs: Graph) -> Graph:
    merged = Graph()
    for graph in graphs:
        for triple in graph:
            merged.add(triple)
    return merged


def compute_expanded_stats(graph: Graph) -> dict:
    predicates = {str(p) for _, p, _ in graph}
    entities = {str(s) for s, _, _ in graph if isinstance(s, URIRef)} | {
        str(o) for _, _, o in graph if isinstance(o, URIRef)
    }
    return {
        "total_triples": len(graph),
        "total_entities": len(entities),
        "total_relations": len(predicates),
    }


def build_alignment_and_expansion(triples_df: pd.DataFrame, initial_kg: Graph) -> AlignmentArtifacts:
    artifacts = build_kg_artifacts(triples_df)
    important_df = select_important_entities(artifacts.cleaned_triples)
    alignment_df = align_entities(important_df)
    alignment_graph = build_alignment_graph(alignment_df)
    predicate_alignment_graph = align_predicates()

    accepted_seed_uris = alignment_df.loc[
        alignment_df["decision"] == "accept_alignment", "external_uri"
    ].dropna().tolist()

    first_hop = fetch_one_hop_expansion(accepted_seed_uris, limit=50000)
    first_hop_objects = sorted({o for _, _, o in first_hop})
    second_hop = fetch_second_hop_expansion(first_hop_objects, limit=120000, sample_size=1800)

    external_graph = triples_to_graph(first_hop + second_hop)
    external_graph = filter_external_graph(external_graph, max_predicates=125)
    expanded_graph = merge_graphs(initial_kg, alignment_graph, predicate_alignment_graph, external_graph)
    expanded_stats = compute_expanded_stats(expanded_graph)

    return AlignmentArtifacts(
        alignment_df=alignment_df,
        alignment_graph=alignment_graph,
        predicate_alignment_graph=predicate_alignment_graph,
        expanded_graph=expanded_graph,
        expanded_stats=expanded_stats,
    )
