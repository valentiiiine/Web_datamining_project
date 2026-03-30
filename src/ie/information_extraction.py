from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span


TARGET_ENTITY_TYPES = {"PERSON", "ORG", "GPE", "DATE"}
GENERIC_ENTITY_TERMS = {
    "open",
    "final",
    "championship",
    "championships",
    "seed",
    "court",
}
GENERIC_RELATIONS = {"be", "have", "do", "say"}
NOISY_ENTITY_PATTERNS = [
    r"^\s*[-*•]+\s*",
    r"^archived from\b",
    r"^retrieved\b",
]

ENTITY_ALIASES = {
    "roland-garros": "French Open",
    "roland garros": "French Open",
    "wimbledon": "Wimbledon Championships",
    "u.s. open": "US Open",
    "us open": "US Open",
    "australian open": "Australian Open",
    "french open": "French Open",
    "novak djokovic": "Novak Djokovic",
    "rafael nadal": "Rafael Nadal",
    "carlos alcaraz": "Carlos Alcaraz",
}

RELATION_KEYWORDS = {
    "won": "won",
    "defeated": "playedAgainst",
    "beat": "playedAgainst",
    "lost to": "playedAgainst",
    "played against": "playedAgainst",
    "played": "participatedIn",
    "participated": "participatedIn",
    "competed": "participatedIn",
    "surface": "hasSurface",
    "clay": "hasSurface",
    "grass": "hasSurface",
    "hard": "hasSurface",
    "from": "fromCountry",
}

SURFACE_TERMS = {"clay", "grass", "hard court", "hard"}
ALLOWED_CANONICAL_RELATIONS = {
    "won",
    "playedAgainst",
    "participatedIn",
    "hasSurface",
    "fromCountry",
    "editionYear",
}
TOURNAMENT_HINTS = {
    "australian open",
    "french open",
    "wimbledon championships",
    "us open",
    "grand slam",
    "roland-garros",
    "roland garros",
}
KNOWN_TOURNAMENT_ALIASES = {
    "Australian Open": ["australian open"],
    "French Open": ["french open", "roland-garros", "roland garros"],
    "Wimbledon Championships": ["wimbledon", "wimbledon championships"],
    "US Open": ["u.s. open", "us open", "us open championships"],
    "Grand Slam": ["grand slam"],
}


@dataclass
class EntityRecord:
    text: str
    label: str
    normalized_text: str
    source_url: str
    sentence: str

    def to_dict(self) -> dict:
        return {
            "entity_text": self.text,
            "entity_label": self.label,
            "normalized_entity": self.normalized_text,
            "source_url": self.source_url,
            "sentence": self.sentence,
        }


def load_spacy_model(model_name: str = "en_core_web_trf") -> Language:
    return spacy.load(model_name)


def normalize_entity_name(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    alias = ENTITY_ALIASES.get(cleaned.lower())
    if alias:
        return alias
    return cleaned


def is_generic_entity(text: str) -> bool:
    cleaned = normalize_entity_name(text).strip()
    if len(cleaned) <= 2:
        return True
    return cleaned.lower() in GENERIC_ENTITY_TERMS


def clean_entity_text(text: str) -> str:
    cleaned = normalize_entity_name(text)
    for pattern in NOISY_ENTITY_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -•*.\t\n\r")
    return cleaned


def is_noisy_entity_text(text: str) -> bool:
    if not text:
        return True
    if len(text) < 3:
        return True
    if text.isdigit():
        return True
    if text.isupper() and len(re.sub(r"[^A-Za-z]", "", text)) >= 3:
        return True
    lowered = text.lower()
    if lowered.startswith("archived from"):
        return True
    if text.startswith(("-", "*", "•")):
        return True
    return False


def is_meaningful_entity(span: Span) -> bool:
    if span.label_ not in TARGET_ENTITY_TYPES:
        return False
    text = clean_entity_text(span.text)
    if not text:
        return False
    if is_generic_entity(text):
        return False
    if is_noisy_entity_text(text):
        return False
    if len(re.sub(r"[^A-Za-z0-9]", "", text)) < 3:
        return False
    return True


def infer_entity_role(entity_text: str, entity_label: str) -> str:
    lowered = entity_text.lower()
    if entity_label == "PERSON":
        return "Player"
    if entity_label == "GPE":
        return "Country"
    if entity_label == "DATE":
        return "Year"
    if lowered in SURFACE_TERMS:
        return "Surface"
    if any(hint in lowered for hint in TOURNAMENT_HINTS):
        return "Tournament"
    if entity_label == "ORG" and any(token in lowered for token in {"open", "wimbledon", "championship", "slam"}):
        return "Tournament"
    return "Other"


def chunk_text(text: str, max_chars: int = 3500) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(buffer) + len(paragraph) + 1 <= max_chars:
            buffer = f"{buffer}\n{paragraph}".strip()
        else:
            if buffer:
                chunks.append(buffer)
            if len(paragraph) <= max_chars:
                buffer = paragraph
            else:
                for start in range(0, len(paragraph), max_chars):
                    chunks.append(paragraph[start : start + max_chars])
                buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks


def detect_tournaments_in_sentence(sentence_text: str) -> list[str]:
    sentence_lower = sentence_text.lower()
    matches: list[str] = []
    for canonical_name, aliases in KNOWN_TOURNAMENT_ALIASES.items():
        if any(alias in sentence_lower for alias in aliases):
            matches.append(canonical_name)
    return matches


def extract_entities_from_doc(doc: Doc, source_url: str) -> list[EntityRecord]:
    records: list[EntityRecord] = []
    for sent in doc.sents:
        for ent in sent.ents:
            if not is_meaningful_entity(ent):
                continue
            cleaned_text = clean_entity_text(ent.text)
            records.append(
                EntityRecord(
                    text=cleaned_text,
                    label=ent.label_,
                    normalized_text=normalize_entity_name(cleaned_text),
                    source_url=source_url,
                    sentence=sent.text.strip(),
                )
            )
    return records


def sentence_relation(sent: Span, entity_names: list[str]) -> str | None:
    lowered = sent.text.lower()
    for keyword, relation in RELATION_KEYWORDS.items():
        if keyword in lowered:
            return relation

    normalized_entities = {name.lower() for name in entity_names}
    if normalized_entities & SURFACE_TERMS:
        return "hasSurface"
    return None


def dependency_relation(sent: Span, subject_text: str, object_text: str) -> str | None:
    for token in sent:
        if token.dep_ not in {"ROOT", "attr", "pobj"}:
            continue
        lemma = token.lemma_.lower().strip()
        if lemma in GENERIC_RELATIONS:
            continue
        relation_map = {
            "win": "won",
            "defeat": "playedAgainst",
            "beat": "playedAgainst",
            "play": "participatedIn",
            "compete": "participatedIn",
            "represent": "fromCountry",
        }
        if lemma not in relation_map:
            continue
        children = {child.dep_: child for child in token.children}
        if "nsubj" in children and "dobj" in children:
            nsubj_text = normalize_entity_name(children["nsubj"].text)
            dobj_text = normalize_entity_name(children["dobj"].text)
            if nsubj_text == subject_text and dobj_text == object_text:
                return relation_map[lemma]
    return None


def is_valid_triple_candidate(
    subject_text: str,
    subject_type: str,
    relation: str,
    object_text: str,
    object_type: str,
) -> bool:
    if relation not in ALLOWED_CANONICAL_RELATIONS:
        return False
    if subject_text == object_text:
        return False
    if not relation.strip():
        return False
    if len(subject_text) < 3 or len(object_text) < 3:
        return False

    subject_role = infer_entity_role(subject_text, subject_type)
    object_role = infer_entity_role(object_text, object_type)

    allowed_patterns = {
        ("Player", "won", "Tournament"),
        ("Player", "playedAgainst", "Player"),
        ("Player", "participatedIn", "Tournament"),
        ("Tournament", "hasSurface", "Surface"),
        ("Player", "fromCountry", "Country"),
        ("Tournament", "editionYear", "Year"),
    }

    if (subject_role, relation, object_role) in allowed_patterns:
        return True

    if relation == "hasSurface" and subject_role == "Tournament" and object_text.lower() in SURFACE_TERMS:
        return True

    if relation == "editionYear" and subject_role == "Tournament" and object_type == "DATE":
        return True

    return False


def extract_candidate_triples_from_doc(doc: Doc, source_url: str) -> list[dict]:
    triples: list[dict] = []
    for sent in doc.sents:
        entities = [ent for ent in sent.ents if is_meaningful_entity(ent)]
        if not entities:
            continue

        sentence_text = sent.text.strip()
        sentence_lower = sentence_text.lower()
        if len(sentence_text) > 400:
            continue
        if "replaced by" in sentence_lower or "archived from the original" in sentence_lower:
            continue
        if sentence_text.count(" - ") > 2 or sentence_text.count("→") > 0:
            continue
        persons = [ent for ent in entities if ent.label_ == "PERSON"]
        countries = [ent for ent in entities if ent.label_ == "GPE"]
        orgs = [ent for ent in entities if infer_entity_role(normalize_entity_name(ent.text), ent.label_) == "Tournament"]
        detected_tournaments = detect_tournaments_in_sentence(sentence_text)

        # Strategy 1: dependency-oriented extraction for opponent relations.
        if any(keyword in sentence_lower for keyword in ["defeated", "beat", "lost to", "played against"]):
            if len(persons) > 3:
                continue
            for i, subj in enumerate(persons):
                for j, obj in enumerate(persons):
                    if i >= j:
                        continue
                    subject_text = normalize_entity_name(subj.text)
                    object_text = normalize_entity_name(obj.text)
                    relation = "playedAgainst"
                    if not is_valid_triple_candidate(subject_text, subj.label_, relation, object_text, obj.label_):
                        continue
                    triples.append(
                        {
                            "subject": subject_text,
                            "subject_type": subj.label_,
                            "subject_role": infer_entity_role(subject_text, subj.label_),
                            "relation": relation,
                            "object": object_text,
                            "object_type": obj.label_,
                            "object_role": infer_entity_role(object_text, obj.label_),
                            "sentence": sentence_text,
                            "source_url": source_url,
                            "strategy": "dependency",
                        }
                    )

        # Strategy 2: sentence-level co-occurrence with tennis-specific constraints.
        if "won" in sentence_lower:
            for subj in persons:
                subject_text = normalize_entity_name(subj.text)
                tournament_objects = [normalize_entity_name(obj.text) for obj in orgs]
                tournament_objects.extend(detected_tournaments)
                for object_text in sorted(set(tournament_objects)):
                    relation = "won"
                    if not is_valid_triple_candidate(subject_text, subj.label_, relation, object_text, "ORG"):
                        continue
                    triples.append(
                        {
                            "subject": subject_text,
                            "subject_type": subj.label_,
                            "subject_role": infer_entity_role(subject_text, subj.label_),
                            "relation": relation,
                            "object": object_text,
                            "object_type": "ORG",
                            "object_role": "Tournament",
                            "sentence": sentence_text,
                            "source_url": source_url,
                            "strategy": "cooccurrence",
                        }
                    )

        if any(keyword in sentence_lower for keyword in ["participated", "played", "competed"]):
            for subj in persons:
                subject_text = normalize_entity_name(subj.text)
                tournament_objects = [normalize_entity_name(obj.text) for obj in orgs]
                tournament_objects.extend(detected_tournaments)
                for object_text in sorted(set(tournament_objects)):
                    relation = "participatedIn"
                    if not is_valid_triple_candidate(subject_text, subj.label_, relation, object_text, "ORG"):
                        continue
                    triples.append(
                        {
                            "subject": subject_text,
                            "subject_type": subj.label_,
                            "subject_role": infer_entity_role(subject_text, subj.label_),
                            "relation": relation,
                            "object": object_text,
                            "object_type": "ORG",
                            "object_role": "Tournament",
                            "sentence": sentence_text,
                            "source_url": source_url,
                            "strategy": "cooccurrence",
                        }
                    )

        if any(keyword in sentence_lower for keyword in ["from", "representing", "nationality"]):
            for subj in persons:
                subject_text = normalize_entity_name(subj.text)
                for obj in countries:
                    object_text = normalize_entity_name(obj.text)
                    if any(hint in object_text.lower() for hint in TOURNAMENT_HINTS):
                        continue
                    if not re.search(
                        rf"{re.escape(subj.text)}.*\bfrom\b.*{re.escape(obj.text)}",
                        sentence_text,
                        flags=re.IGNORECASE,
                    ):
                        continue
                    relation = "fromCountry"
                    if not is_valid_triple_candidate(subject_text, subj.label_, relation, object_text, obj.label_):
                        continue
                    triples.append(
                        {
                            "subject": subject_text,
                            "subject_type": subj.label_,
                            "subject_role": infer_entity_role(subject_text, subj.label_),
                            "relation": relation,
                            "object": object_text,
                            "object_type": obj.label_,
                            "object_role": infer_entity_role(object_text, obj.label_),
                            "sentence": sentence_text,
                            "source_url": source_url,
                            "strategy": "cooccurrence",
                        }
                    )

        if any(keyword in sentence_lower for keyword in ["surface", "clay", "grass", "hard court", "hard courts"]):
            surface_matches = []
            for term in SURFACE_TERMS:
                if term in sentence_lower:
                    surface_matches.append(term.title() if term != "hard" else "Hard")
            subject_candidates = [normalize_entity_name(subj.text) for subj in orgs]
            subject_candidates.extend(detected_tournaments)
            for subject_text in sorted(set(subject_candidates)):
                if not any(hint in subject_text.lower() for hint in TOURNAMENT_HINTS):
                    continue
                for surface in surface_matches:
                    relation = "hasSurface"
                    if not is_valid_triple_candidate(subject_text, "ORG", relation, surface, "ORG"):
                        continue
                    triples.append(
                        {
                            "subject": subject_text,
                            "subject_type": "ORG",
                            "subject_role": "Tournament",
                            "relation": relation,
                            "object": surface,
                            "object_type": "ORG",
                            "object_role": "Surface",
                            "sentence": sentence_text,
                            "source_url": source_url,
                            "strategy": "cooccurrence",
                        }
                    )
    return triples


def process_corpus(records: Iterable[dict], nlp: Language, max_docs: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    entity_rows: list[dict] = []
    triple_rows: list[dict] = []

    selected_records = list(records)
    if max_docs is not None:
        selected_records = selected_records[:max_docs]

    texts: list[str] = []
    metadata: list[str] = []
    for record in selected_records:
        chunks = chunk_text(record["text"])
        texts.extend(chunks)
        metadata.extend([record["url"]] * len(chunks))

    for doc, source_url in zip(nlp.pipe(texts, batch_size=4), metadata):
        entity_rows.extend(item.to_dict() for item in extract_entities_from_doc(doc, source_url))
        triple_rows.extend(extract_candidate_triples_from_doc(doc, source_url))

    entities_df = pd.DataFrame(entity_rows).drop_duplicates()
    triples_df = pd.DataFrame(triple_rows).drop_duplicates()

    if not entities_df.empty:
        # Cleaning immediately after NER is critical because noisy nodes
        # propagate into KG construction, alignment, and RAG evaluation.
        entities_df["entity_text"] = entities_df["entity_text"].map(clean_entity_text)
        entities_df["normalized_entity"] = entities_df["entity_text"].map(normalize_entity_name)
        text_mask = entities_df["entity_text"].map(lambda value: not is_noisy_entity_text(str(value)))
        entities_df = entities_df[text_mask]
        entities_df = entities_df.sort_values(["normalized_entity", "entity_label", "source_url"]).reset_index(drop=True)
        entities_df = entities_df.drop_duplicates(
            subset=["normalized_entity", "entity_label", "source_url", "sentence"]
        ).reset_index(drop=True)
    if not triples_df.empty:
        triples_df = triples_df.sort_values(["subject", "relation", "object", "source_url"]).reset_index(drop=True)

    return entities_df, triples_df


def compute_entity_type_distribution(entities_df: pd.DataFrame) -> pd.DataFrame:
    counts = entities_df["entity_label"].value_counts().rename_axis("entity_type").reset_index(name="count")
    return counts


def sample_noisy_entities(entities_df: pd.DataFrame) -> list[str]:
    examples = []
    seen = set()
    for entity in entities_df["entity_text"].tolist():
        if is_generic_entity(entity) and entity not in seen:
            examples.append(entity)
            seen.add(entity)
    return examples[:10]


def noisy_triple_ratio(triples_df: pd.DataFrame, sample_size: int = 30) -> float:
    if triples_df.empty:
        return 0.0
    sample = triples_df.head(sample_size)
    noisy = 0
    for _, row in sample.iterrows():
        if row["subject"] == row["object"]:
            noisy += 1
        elif not row["relation"] or row["relation"] in GENERIC_RELATIONS:
            noisy += 1
        elif len(row["subject"]) < 3 or len(row["object"]) < 3:
            noisy += 1
    return noisy / len(sample)


def relation_distribution(triples_df: pd.DataFrame) -> pd.DataFrame:
    if triples_df.empty:
        return pd.DataFrame(columns=["relation", "count"])
    return triples_df["relation"].value_counts().rename_axis("relation").reset_index(name="count")


def manual_review_sample(triples_df: pd.DataFrame, sample_size: int = 10) -> pd.DataFrame:
    if triples_df.empty:
        return triples_df
    return triples_df.head(sample_size).copy()


def count_filtered_generic_candidates(raw_entities: Iterable[str]) -> Counter:
    counter = Counter()
    for entity in raw_entities:
        if is_generic_entity(entity):
            counter[normalize_entity_name(entity)] += 1
    return counter
