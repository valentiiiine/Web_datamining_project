from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, URIRef, XSD


EX = Namespace("http://example.org/tennis/")
ONTO = Namespace("http://example.org/tennis/ontology/")
PROV = Namespace("http://example.org/tennis/provenance/")

CLASS_URIS = {
    "Player": ONTO.Player,
    "Tournament": ONTO.Tournament,
    "Match": ONTO.Match,
    "Surface": ONTO.Surface,
    "Country": ONTO.Country,
    "TournamentEdition": ONTO.TournamentEdition,
}

PROPERTY_URIS = {
    "won": ONTO.won,
    "participatedIn": ONTO.participatedIn,
    "playedAgainst": ONTO.playedAgainst,
    "hasSurface": ONTO.hasSurface,
    "fromCountry": ONTO.fromCountry,
    "editionYear": ONTO.editionYear,
    "sourceURL": PROV.sourceURL,
    "evidenceText": PROV.evidenceText,
}

RELATION_DOMAINS_RANGES = {
    "won": ("Player", "Tournament"),
    "participatedIn": ("Player", "Tournament"),
    "playedAgainst": ("Player", "Player"),
    "hasSurface": ("Tournament", "Surface"),
    "fromCountry": ("Player", "Country"),
    "editionYear": ("TournamentEdition", "YearLiteral"),
}

SURFACE_NORMALIZATION = {
    "hard": "Hard Court",
    "hard court": "Hard Court",
    "grass": "Grass",
    "clay": "Clay",
}

PLAYER_ALIASES = {
    "nadal": "Rafael Nadal",
    "rafa": "Rafael Nadal",
    "djokovic": "Novak Djokovic",
    "novak": "Novak Djokovic",
    "alcaraz": "Carlos Alcaraz",
    "agassi": "Andre Agassi",
    "federer": "Roger Federer",
    "murray": "Andy Murray",
    "sinner": "Jannik Sinner",
    "tsitsipas": "Stefanos Tsitsipas",
    "zverev": "Alexander Zverev",
    "wawrinka": "Stan Wawrinka",
    "ferrer": "David Ferrer",
}

TOURNAMENT_ALIASES = {
    "wimbledon": "Wimbledon Championships",
    "roland garros": "French Open",
    "roland-garros": "French Open",
    "u.s. open": "US Open",
    "us open championships": "US Open",
    "grand slam": "Grand Slam",
}

BAD_ENTITY_TERMS = {
    "open era",
    "grand slam",
    "the final",
    "final",
    "championship",
    "championships",
}


@dataclass
class KGArtifacts:
    ontology_graph: Graph
    instance_graph: Graph
    cleaned_triples: pd.DataFrame
    stats: dict


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", ascii_text).strip("_")
    return cleaned or "unnamed"


def normalize_label(text: str, role: str | None = None) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    lowered = cleaned.lower()
    if role == "Surface":
        return SURFACE_NORMALIZATION.get(lowered, cleaned)
    if role == "Player":
        return PLAYER_ALIASES.get(lowered, cleaned)
    if role == "Tournament":
        return TOURNAMENT_ALIASES.get(lowered, cleaned)
    if role == "Country":
        cleaned = re.sub(r"\[\d+\]", "", cleaned).strip(" .")
        return cleaned
    if lowered in PLAYER_ALIASES:
        return PLAYER_ALIASES[lowered]
    if lowered in TOURNAMENT_ALIASES:
        return TOURNAMENT_ALIASES[lowered]
    return cleaned


def role_to_uri(role: str, label: str) -> URIRef:
    category = role.lower()
    return EX[f"{category}/{slugify(label)}"]


def is_meaningful_node(label: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(label)).strip()
    if len(cleaned) < 3:
        return False
    if cleaned.lower() in BAD_ENTITY_TERMS:
        return False
    if cleaned.isdigit() and len(cleaned) < 4:
        return False
    return True


def infer_tournament_edition(sentence: str, tournament_label: str) -> tuple[str | None, int | None]:
    year_match = re.search(r"\b(19|20)\d{2}\b", sentence)
    if not year_match:
        return None, None
    year = int(year_match.group(0))
    return f"{tournament_label} {year}", year


def clean_candidate_triples(triples_df: pd.DataFrame) -> pd.DataFrame:
    cleaned = triples_df.copy()
    cleaned["subject"] = cleaned.apply(
        lambda row: normalize_label(row["subject"], row.get("subject_role")), axis=1
    )
    cleaned["object"] = cleaned.apply(
        lambda row: normalize_label(row["object"], row.get("object_role")), axis=1
    )

    cleaned = cleaned[cleaned["subject"].map(is_meaningful_node)]
    cleaned = cleaned[cleaned["object"].map(is_meaningful_node)]
    cleaned = cleaned[cleaned["relation"].isin({"won", "participatedIn", "playedAgainst", "hasSurface", "fromCountry"})]
    cleaned = cleaned[cleaned["subject"] != cleaned["object"]]

    cleaned = cleaned[
        ~cleaned["sentence"].str.contains(
            r"archived from the original|nobody's idea|Open Era|record of consecutive|queue card",
            case=False,
            regex=True,
            na=False,
        )
    ]

    cleaned = cleaned[
        ~(
            (cleaned["relation"] == "won")
            & (~cleaned["object"].str.contains(r"Open|Wimbledon|French|Grand Slam", case=False, regex=True, na=False))
        )
    ]

    cleaned = cleaned[
        ~(
            (cleaned["relation"] == "fromCountry")
            & (~cleaned["object"].str.fullmatch(r"[A-Za-z .'-]+", na=False))
        )
    ]

    cleaned = cleaned.drop_duplicates(
        subset=["subject", "relation", "object", "sentence", "source_url"]
    ).reset_index(drop=True)
    return cleaned


def build_ontology_graph() -> Graph:
    g = Graph()
    bind_namespaces(g)

    for class_name, class_uri in CLASS_URIS.items():
        g.add((class_uri, RDF.type, OWL.Class))
        g.add((class_uri, RDFS.label, Literal(class_name)))

    for relation, prop_uri in PROPERTY_URIS.items():
        g.add((prop_uri, RDF.type, OWL.ObjectProperty if relation != "editionYear" and relation not in {"sourceURL", "evidenceText"} else OWL.DatatypeProperty))
        g.add((prop_uri, RDFS.label, Literal(relation)))

    for relation, (domain_name, range_name) in RELATION_DOMAINS_RANGES.items():
        prop_uri = PROPERTY_URIS[relation]
        g.add((prop_uri, RDFS.domain, CLASS_URIS[domain_name]))
        if range_name == "YearLiteral":
            g.add((prop_uri, RDFS.range, XSD.gYear))
        else:
            g.add((prop_uri, RDFS.range, CLASS_URIS[range_name]))

    g.add((PROPERTY_URIS["playedAgainst"], RDF.type, OWL.SymmetricProperty))
    return g


def bind_namespaces(g: Graph) -> None:
    g.bind("ex", EX)
    g.bind("onto", ONTO)
    g.bind("prov", PROV)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)


def add_typed_entity(g: Graph, label: str, role: str) -> URIRef:
    uri = role_to_uri(role, label)
    g.add((uri, RDF.type, CLASS_URIS[role]))
    g.add((uri, RDFS.label, Literal(label)))
    return uri


def add_relation_triple(g: Graph, row: pd.Series) -> None:
    subject_role = row["subject_role"]
    object_role = row["object_role"]
    relation = row["relation"]

    if subject_role not in CLASS_URIS:
        return

    if relation == "hasSurface":
        object_role = "Surface"
    if relation == "fromCountry":
        object_role = "Country"
    if relation in {"won", "participatedIn"}:
        object_role = "Tournament"
    if relation == "playedAgainst":
        object_role = "Player"

    if object_role not in CLASS_URIS:
        return

    subject_uri = add_typed_entity(g, row["subject"], subject_role)
    object_uri = add_typed_entity(g, row["object"], object_role)
    predicate_uri = PROPERTY_URIS[relation]

    g.add((subject_uri, predicate_uri, object_uri))
    g.add((subject_uri, PROV.sourceURL, Literal(row["source_url"], datatype=XSD.anyURI)))
    g.add((subject_uri, PROV.evidenceText, Literal(row["sentence"])))

    if relation == "playedAgainst":
        g.add((object_uri, predicate_uri, subject_uri))

    if relation in {"won", "participatedIn"} and object_role == "Tournament":
        edition_label, edition_year = infer_tournament_edition(row["sentence"], row["object"])
        if edition_label and edition_year:
            edition_uri = add_typed_entity(g, edition_label, "TournamentEdition")
            g.add((edition_uri, ONTO.editionYear, Literal(str(edition_year), datatype=XSD.gYear)))
            g.add((edition_uri, RDFS.seeAlso, object_uri))


def build_instance_graph(cleaned_triples: pd.DataFrame) -> Graph:
    g = Graph()
    bind_namespaces(g)

    for _, row in cleaned_triples.iterrows():
        add_relation_triple(g, row)

    return g


def compute_graph_statistics(g: Graph) -> dict:
    custom_predicates = {
        str(uri)
        for key, uri in PROPERTY_URIS.items()
        if key in {"won", "participatedIn", "playedAgainst", "hasSurface", "fromCountry", "editionYear"}
    }
    predicates = {str(p) for _, p, _ in g if str(p) in custom_predicates}
    subjects = {str(s) for s, _, _ in g if isinstance(s, URIRef)}
    objects = {str(o) for _, _, o in g if isinstance(o, URIRef)}
    entity_uris = subjects | objects

    class_counts = {}
    for class_name, class_uri in CLASS_URIS.items():
        class_counts[class_name] = sum(1 for _ in g.triples((None, RDF.type, class_uri)))

    return {
        "total_triples": len(g),
        "total_entities": len(entity_uris),
        "total_relations": len(predicates),
        "predicate_uris": sorted(predicates),
        "class_counts": class_counts,
    }


def serialize_graph(graph: Graph, output_path: str, fmt: str = "turtle") -> None:
    graph.serialize(destination=output_path, format=fmt)


def build_kg_artifacts(triples_df: pd.DataFrame) -> KGArtifacts:
    cleaned_triples = clean_candidate_triples(triples_df)
    ontology_graph = build_ontology_graph()
    instance_graph = build_instance_graph(cleaned_triples)
    stats = compute_graph_statistics(instance_graph)
    return KGArtifacts(
        ontology_graph=ontology_graph,
        instance_graph=instance_graph,
        cleaned_triples=cleaned_triples,
        stats=stats,
    )
