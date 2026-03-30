from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from owlready2 import Imp, ObjectProperty, DataProperty, Thing, get_ontology, sync_reasoner_pellet
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from rdflib import Graph, Literal, RDF, RDFS, URIRef
from sklearn.manifold import TSNE


LOCAL_ONTOLOGY_PREFIX = "http://example.org/tennis/ontology/"
LOCAL_ENTITY_PREFIX = "http://example.org/tennis/"
WIKIDATA_DIRECT_PREFIX = "http://www.wikidata.org/prop/direct/"
EXCLUDED_PREDICATE_KEYWORDS = {
    "label",
    "description",
    "evidenceText",
    "sourceURL",
    "sameAs",
    "type",
    "seeAlso",
}


def run_family_swrl_reasoning(family_owl_path: str) -> tuple[list[dict], str]:
    onto = get_ontology(Path(family_owl_path).resolve().as_uri()).load()
    with onto:
        rule = Imp()
        rule.set_as_rule(
            "Person(?p), hasAge(?p, ?age), greaterThan(?age, 60) -> OldPerson(?p)"
        )

    sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)

    inferred = []
    for individual in onto.individuals():
        inferred.append(
            {
                "individual": individual.name,
                "types": sorted(cls.name for cls in individual.is_a if hasattr(cls, "name")),
                "age": individual.hasAge[0] if hasattr(individual, "hasAge") and individual.hasAge else None,
                "is_old_person": any(getattr(cls, "name", "") == "OldPerson" for cls in individual.is_a),
            }
        )
    return inferred, "Person(?p), hasAge(?p, ?age), greaterThan(?age, 60) -> OldPerson(?p)"


def build_project_reasoning_ontology(cleaned_triples_df: pd.DataFrame, output_path: str | None = None):
    onto = get_ontology("http://example.org/tennis_reasoning.owl")
    with onto:
        class Player(Thing):
            pass

        class Tournament(Thing):
            pass

        class TopPlayer(Player):
            pass

        class won(ObjectProperty):
            domain = [Player]
            range = [Tournament]

        class hasMajorWins(DataProperty):
            domain = [Player]
            range = [int]

        rule = Imp()
        rule.set_as_rule(
            "Player(?p), hasMajorWins(?p, ?n), greaterThan(?n, 2) -> TopPlayer(?p)"
        )

        player_entities = {}
        tournament_entities = {}

        for player_name in sorted(cleaned_triples_df["subject"].loc[cleaned_triples_df["subject_role"] == "Player"].unique()):
            player_entities[player_name] = Player(player_name.replace(" ", "_"))

        for tournament_name in sorted(cleaned_triples_df["object"].loc[cleaned_triples_df["object_role"] == "Tournament"].unique()):
            tournament_entities[tournament_name] = Tournament(tournament_name.replace(" ", "_"))

        win_counts = (
            cleaned_triples_df.loc[cleaned_triples_df["relation"] == "won", "subject"]
            .value_counts()
            .to_dict()
        )

        for _, row in cleaned_triples_df.loc[cleaned_triples_df["relation"] == "won"].iterrows():
            player_name = row["subject"]
            tournament_name = row["object"]
            if player_name in player_entities and tournament_name in tournament_entities:
                player_entities[player_name].won.append(tournament_entities[tournament_name])

        for player_name, player_individual in player_entities.items():
            player_individual.hasMajorWins = [int(win_counts.get(player_name, 0))]

    sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)

    inferred_top_players = []
    for individual in onto.individuals():
        types = sorted(cls.name for cls in individual.is_a if hasattr(cls, "name"))
        if "TopPlayer" in types:
            inferred_top_players.append(
                {
                    "player": individual.name.replace("_", " "),
                    "types": types,
                    "major_wins_count": individual.hasMajorWins[0] if getattr(individual, "hasMajorWins", None) else None,
                }
            )

    if output_path:
        onto.save(file=output_path, format="rdfxml")

    return (
        inferred_top_players,
        "Player(?p), hasMajorWins(?p, ?n), greaterThan(?n, 2) -> TopPlayer(?p)",
    )


def predicate_is_kge_ready(predicate_uri: str) -> bool:
    lowered = predicate_uri.lower()
    if any(keyword.lower() in lowered for keyword in EXCLUDED_PREDICATE_KEYWORDS):
        return False
    if predicate_uri.startswith("http://www.w3.org/"):
        return False
    if predicate_uri.startswith("http://example.org/tennis/provenance/"):
        return False
    return True


def graph_to_kge_dataframe(graph_path: str) -> pd.DataFrame:
    graph = Graph()
    graph.parse(graph_path, format="turtle")
    rows = []
    for s, p, o in graph:
        if not isinstance(s, URIRef) or not isinstance(o, URIRef):
            continue
        predicate = str(p)
        if not predicate_is_kge_ready(predicate):
            continue
        rows.append((str(s), predicate, str(o)))

    df = pd.DataFrame(rows, columns=["head", "relation", "tail"]).drop_duplicates().reset_index(drop=True)
    return df


def filter_low_frequency_entities(df: pd.DataFrame, min_degree: int = 2) -> pd.DataFrame:
    degree = Counter(df["head"]) + Counter(df["tail"])
    filtered = df[df["head"].map(degree.get) >= min_degree]
    filtered = filtered[filtered["tail"].map(degree.get) >= min_degree]
    return filtered.reset_index(drop=True)


def make_subset(df: pd.DataFrame, size: int | None, random_state: int = 42) -> pd.DataFrame:
    if size is None or size >= len(df):
        return df.copy().reset_index(drop=True)
    sampled = df.sample(n=size, random_state=random_state).reset_index(drop=True)
    return sampled


def split_without_unseen_entities(df: pd.DataFrame, random_state: int = 42):
    tf = dataframe_to_triples_factory(df)
    train_tf, valid_tf, test_tf = tf.split(
        ratios=[0.8, 0.1, 0.1],
        random_state=random_state,
        method="cleanup",
    )

    def triples_factory_to_df(factory: TriplesFactory) -> pd.DataFrame:
        id_to_entity = {idx: label for label, idx in factory.entity_to_id.items()}
        id_to_relation = {idx: label for label, idx in factory.relation_to_id.items()}
        triples = []
        for h, r, t in factory.mapped_triples.cpu().numpy():
            triples.append((id_to_entity[int(h)], id_to_relation[int(r)], id_to_entity[int(t)]))
        return pd.DataFrame(triples, columns=["head", "relation", "tail"]).drop_duplicates().reset_index(drop=True)

    return triples_factory_to_df(train_tf), triples_factory_to_df(valid_tf), triples_factory_to_df(test_tf)


def save_split_files(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, df in [("train.txt", train_df), ("valid.txt", valid_df), ("test.txt", test_df)]:
        df.to_csv(output / name, sep="\t", index=False, header=False)


def dataframe_to_triples_factory(df: pd.DataFrame) -> TriplesFactory:
    triples = df[["head", "relation", "tail"]].astype(str).to_numpy()
    return TriplesFactory.from_labeled_triples(triples)


def train_and_evaluate_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    random_state: int = 42,
    embedding_dim: int = 64,
    learning_rate: float = 0.01,
    batch_size: int = 256,
    epochs: int = 8,
):
    train_tf = dataframe_to_triples_factory(train_df)
    valid_tf = TriplesFactory.from_labeled_triples(
        valid_df[["head", "relation", "tail"]].astype(str).to_numpy(),
        entity_to_id=train_tf.entity_to_id,
        relation_to_id=train_tf.relation_to_id,
    )
    test_tf = TriplesFactory.from_labeled_triples(
        test_df[["head", "relation", "tail"]].astype(str).to_numpy(),
        entity_to_id=train_tf.entity_to_id,
        relation_to_id=train_tf.relation_to_id,
    )

    result = pipeline(
        training=train_tf,
        validation=valid_tf,
        testing=test_tf,
        model=model_name,
        random_seed=random_state,
        model_kwargs={"embedding_dim": embedding_dim},
        optimizer_kwargs={"lr": learning_rate},
        training_kwargs={"num_epochs": epochs, "batch_size": batch_size},
        negative_sampler="basic",
        evaluator_kwargs={"filtered": True},
        stopper="early",
        stopper_kwargs={"frequency": 2, "patience": 2, "relative_delta": 0.002},
        device="cpu",
    )

    metric_results = result.metric_results.to_dict()
    summary = {
        "model": model_name,
        "embedding_dim": embedding_dim,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "MRR": metric_results["both"]["realistic"]["inverse_harmonic_mean_rank"],
        "Hits@1": metric_results["both"]["realistic"]["hits_at_1"],
        "Hits@3": metric_results["both"]["realistic"]["hits_at_3"],
        "Hits@10": metric_results["both"]["realistic"]["hits_at_10"],
    }
    return result, summary


def nearest_neighbors_from_model(result, entity_label: str, top_k: int = 5):
    model = result.model
    entity_to_id = result.training.entity_to_id
    if entity_label not in entity_to_id:
        return []

    entity_id = entity_to_id[entity_label]
    embeddings = model.entity_representations[0]._embeddings.weight.detach().cpu().numpy()
    target_vector = embeddings[entity_id]
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(target_vector)
    similarities = np.dot(embeddings, target_vector) / np.where(norms == 0, 1e-9, norms)

    id_to_entity = {idx: label for label, idx in entity_to_id.items()}
    ranked_ids = np.argsort(-similarities)
    neighbors = []
    for idx in ranked_ids:
        if idx == entity_id:
            continue
        neighbors.append((id_to_entity[idx], float(similarities[idx])))
        if len(neighbors) >= top_k:
            break
    return neighbors


def tsne_projection_from_model(result, max_points: int = 300, random_state: int = 42) -> pd.DataFrame:
    model = result.model
    embeddings = model.entity_representations[0]._embeddings.weight.detach().cpu().numpy()
    entity_labels = list(result.training.entity_to_id.keys())

    if len(entity_labels) > max_points:
        rng = np.random.default_rng(random_state)
        indices = np.sort(rng.choice(len(entity_labels), size=max_points, replace=False))
        embeddings = embeddings[indices]
        entity_labels = [entity_labels[i] for i in indices]

    projection = TSNE(n_components=2, random_state=random_state, perplexity=min(30, max(5, len(entity_labels) // 8))).fit_transform(embeddings)
    return pd.DataFrame({"entity": entity_labels, "x": projection[:, 0], "y": projection[:, 1]})


def plot_tsne(tsne_df: pd.DataFrame, output_path: str | None = None) -> None:
    plt.figure(figsize=(10, 7))
    plt.scatter(tsne_df["x"], tsne_df["y"], alpha=0.7, s=25)
    for _, row in tsne_df.head(30).iterrows():
        plt.text(row["x"], row["y"], row["entity"].split("/")[-1][:20], fontsize=8)
    plt.title("t-SNE projection of entity embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200)
    plt.close()


def evaluate_size_experiments(
    kge_df: pd.DataFrame,
    output_root: str,
    model_names: list[str] | None = None,
    size_plan: list[tuple[str, int | None]] | None = None,
    random_state: int = 42,
    embedding_dim: int = 64,
    learning_rate: float = 0.01,
    batch_size: int = 256,
    epochs: int = 4,
):
    if model_names is None:
        model_names = ["TransE", "DistMult"]
    if size_plan is None:
        size_plan = [("20k", 20000), ("50k", 50000), ("full", None)]

    all_results = []
    trained_models = {}

    for size_label, size_value in size_plan:
        subset_df = make_subset(kge_df, size_value, random_state=random_state)
        train_df, valid_df, test_df = split_without_unseen_entities(subset_df, random_state=random_state)
        split_dir = Path(output_root) / size_label
        save_split_files(train_df, valid_df, test_df, str(split_dir))

        for model_name in model_names:
            result, summary = train_and_evaluate_model(
                train_df=train_df,
                valid_df=valid_df,
                test_df=test_df,
                model_name=model_name,
                random_state=random_state,
                embedding_dim=embedding_dim,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
            )
            summary["dataset_size"] = size_label
            summary["train_triples"] = len(train_df)
            summary["valid_triples"] = len(valid_df)
            summary["test_triples"] = len(test_df)
            all_results.append(summary)
            trained_models[(size_label, model_name)] = {
                "result": result,
                "train_df": train_df,
                "valid_df": valid_df,
                "test_df": test_df,
            }

    results_df = pd.DataFrame(all_results)
    return results_df, trained_models
