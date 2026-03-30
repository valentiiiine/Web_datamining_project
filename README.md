# Tennis Grand Slam Knowledge Graph and Intelligent QA System

This repository contains a complete academic pipeline for building a tennis Grand Slam knowledge graph and a grounded question answering system based on RDF, SPARQL, SWRL reasoning, knowledge graph embeddings, and a local LLM.

The project follows the required progression:

1. Web data collection and cleaning
2. Information extraction
3. RDF knowledge graph construction
4. Alignment with Wikidata and KB expansion
5. SWRL reasoning and KGE experiments
6. SPARQL-generation RAG with a local Ollama model

## Repository Structure

```text
project-root/
├─ src/
│  ├─ crawl/
│  ├─ ie/
│  ├─ kg/
│  ├─ kge/
│  ├─ rag/
│  └─ reason/
├─ data/
│  ├─ raw/
│  ├─ interim/
│  ├─ kge/
│  └─ samples/
├─ kg_artifacts/
│  ├─ ontology.ttl
│  ├─ initial_kg.ttl
│  ├─ expanded_kg.ttl
│  ├─ rag_tennis_kg.ttl
│  ├─ alignment.ttl
│  └─ tennis_reasoning.owl
├─ notebooks/
│  ├─ 01_data_collection.ipynb
│  ├─ 02_information_extraction.ipynb
│  ├─ 03_kg_construction.ipynb
│  ├─ 04_alignment_and_expansion.ipynb
│  ├─ 05_reasoning_and_kge.ipynb
│  └─ 06_rag_chatbot.ipynb
├─ reports/
│  └─ final_report.pdf
├─ README.md
└─ requirements.txt
```

## Python and Environment

- Recommended Python version: `3.10` to `3.12`
- Minimum supported version: `>=3.9`
- A virtual environment is strongly recommended

Create and activate an environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

If `pykeen` or `torch` installation is slow on your machine, install PyTorch first using the official instructions for your platform, then run `pip install -r requirements.txt`.

## Main Dependencies

- `requests`, `httpx`
- `trafilatura`
- `spacy`, `en_core_web_trf`
- `pandas`
- `rdflib`
- `owlready2`
- `pykeen`
- `matplotlib`
- `scikit-learn`
- `jupyter`, `nbformat`

## How to Run the Project

Run the notebooks in order:

1. [01_data_collection.ipynb](/Users/vincentlemeur/Documents/S8/DIA/Datamining/Project_datamining_bis/notebooks/01_data_collection.ipynb)
2. [02_information_extraction.ipynb](/Users/vincentlemeur/Documents/S8/DIA/Datamining/Project_datamining_bis/notebooks/02_information_extraction.ipynb)
3. [03_kg_construction.ipynb](/Users/vincentlemeur/Documents/S8/DIA/Datamining/Project_datamining_bis/notebooks/03_kg_construction.ipynb)
4. [04_alignment_and_expansion.ipynb](/Users/vincentlemeur/Documents/S8/DIA/Datamining/Project_datamining_bis/notebooks/04_alignment_and_expansion.ipynb)
5. [05_reasoning_and_kge.ipynb](/Users/vincentlemeur/Documents/S8/DIA/Datamining/Project_datamining_bis/notebooks/05_reasoning_and_kge.ipynb)
6. [06_rag_chatbot.ipynb](/Users/vincentlemeur/Documents/S8/DIA/Datamining/Project_datamining_bis/notebooks/06_rag_chatbot.ipynb)

You can also execute a notebook from the terminal:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/06_rag_chatbot.ipynb
```

## How to Run the KGE Experiments

The KGE workflow is documented in notebook 5 and uses:

- [kg_artifacts/expanded_kg.ttl](/Users/vincentlemeur/Documents/S8/DIA/Datamining/Project_datamining_bis/kg_artifacts/expanded_kg.ttl)
- [data/kge/20k/train.txt](/Users/vincentlemeur/Documents/S8/DIA/Datamining/Project_datamining_bis/data/kge/20k/train.txt)
- [data/kge/50k/train.txt](/Users/vincentlemeur/Documents/S8/DIA/Datamining/Project_datamining_bis/data/kge/50k/train.txt)
- [data/kge/full/train.txt](/Users/vincentlemeur/Documents/S8/DIA/Datamining/Project_datamining_bis/data/kge/full/train.txt)

Outputs are stored in:

- [kg_artifacts/kge_results.csv](/Users/vincentlemeur/Documents/S8/DIA/Datamining/Project_datamining_bis/kg_artifacts/kge_results.csv)
- [kg_artifacts/best_model_summary.json](/Users/vincentlemeur/Documents/S8/DIA/Datamining/Project_datamining_bis/kg_artifacts/best_model_summary.json)
- [kg_artifacts/tsne_plot.png](/Users/vincentlemeur/Documents/S8/DIA/Datamining/Project_datamining_bis/kg_artifacts/tsne_plot.png)

## Local Ollama Setup for the RAG Demo

The RAG part uses a local Ollama service.

1. Start Ollama. The HTTP server should be available at [http://localhost:11434](http://localhost:11434).
2. Open the URL in a browser. A healthy setup returns: `Ollama is running`
3. Pull or run a local model, for example:

```bash
ollama run gemma:2b
```

You may also use:

- `gemma3:1b`
- `gemma2:2b`
- `deepseek-r1:1.5b`
- `qwen` small models
- `llama3.1:8b`

4. Test the API:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "gemma:2b",
  "prompt": "who are you?",
  "format": "json"
}'
```

## How to Run the RAG Demo

The helper script is [lab_rag_sparql_gen.py](/Users/vincentlemeur/Documents/S8/DIA/Datamining/Project_datamining_bis/lab_rag_sparql_gen.py).

Run it with the default model:

```bash
python lab_rag_sparql_gen.py
```

Run it with a stronger local model:

```bash
OLLAMA_MODEL=gemma3:1b python lab_rag_sparql_gen.py
```

Example interaction:

```text
Question (or 'quit'): Which players are from Spain?
--- Baseline (No RAG) ---
...
--- SPARQL-generation RAG (Local LLM + rdflib) ---
[Results]
playerLabel
Carlos Alcaraz
Rafael Nadal
```

## Hardware Notes

- Crawling and IE run comfortably on a laptop
- `en_core_web_trf` benefits from more RAM and can be slower on CPU-only machines
- KGE experiments are manageable on a laptop when using the provided 20k, 50k, and full settings
- Local LLM inference quality improves with stronger models, but `gemma:2b` and `gemma3:1b` are lighter options
- Recommended RAM for a smooth end-to-end run: `16 GB` or more

## Data and Artifacts

Main project outputs:

- [data/raw/crawler_output.jsonl](/Users/vincentlemeur/Documents/S8/DIA/Datamining/Project_datamining_bis/data/raw/crawler_output.jsonl)
- [data/interim/extracted_knowledge.csv](/Users/vincentlemeur/Documents/S8/DIA/Datamining/Project_datamining_bis/data/interim/extracted_knowledge.csv)
- [kg_artifacts/initial_kg.ttl](/Users/vincentlemeur/Documents/S8/DIA/Datamining/Project_datamining_bis/kg_artifacts/initial_kg.ttl)
- [kg_artifacts/expanded_kg.ttl](/Users/vincentlemeur/Documents/S8/DIA/Datamining/Project_datamining_bis/kg_artifacts/expanded_kg.ttl)
- [kg_artifacts/rag_tennis_kg.ttl](/Users/vincentlemeur/Documents/S8/DIA/Datamining/Project_datamining_bis/kg_artifacts/rag_tennis_kg.ttl)

## Final Report Guidance

The report structure template is available in [reports/final_report_structure.md](/Users/vincentlemeur/Documents/S8/DIA/Datamining/Project_datamining_bis/reports/final_report_structure.md).

The report should include:

1. Data acquisition and IE
2. KB construction and alignment
3. SWRL reasoning
4. KGE experiments
5. RAG over RDF/SPARQL
6. Critical reflection

## Screenshot Reminder

Add at least one screenshot before final submission:

- one screenshot of the running RAG CLI or notebook demo
- optionally one screenshot of a plot such as the t-SNE projection

An actual screenshot is not generated automatically in this repository, because it should reflect your real final run on your machine.
