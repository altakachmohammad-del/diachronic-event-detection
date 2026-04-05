# Diachronic Contextual Embeddings for Event Detection

> **M2 Research Internship** — LIASD (Universite Paris 8) & INA (Institut National de l'Audiovisuel)
>
> **Intern:** Mohammed Al-Takach | **Supervisors:** Aurélien Bossard (LIASD) & Abdelkrim Beloued (INA)

## Overview

This project investigates **automatic event detection** through the diachronic analysis of contextual word embeddings of named entities. By tracking how the contextual representations of political figures (e.g., *Macron*) evolve over time in news corpora, we can identify significant events — moments where the semantic context of an entity changes abruptly.

### Key Idea

When the contextual embedding of a named entity **shifts dramatically** between two time periods, it signals that a major event has occurred around that entity.

## Project Structure

```
.
├── src/
│   ├── 01_extract_embeddings.py    # Extract contextual embeddings (BERT/CamemBERT)
│   ├── 02_change_point_detection.py # Detect temporal ruptures (PELT, BinSeg, threshold)
│   └── 03_visualize.py             # Generate analysis plots
├── data/                           # Corpus data (not tracked in git)
├── notebooks/                      # Jupyter notebooks for exploration
├── outputs/                        # Generated results & plots
├── references/                     # Key research papers
│   └── 00_READING_GUIDE.txt        # Prioritized reading list
├── docs/                           # Documentation & reports
├── requirements.txt
└── README.md
```

## Methodology

### 1. Embedding Extraction
- Load a pretrained transformer model (DistilBERT multilingual for local dev, CamemBERT for production)
- For each named entity in each time period, extract the contextual embedding from the last hidden layer
- Pool sub-word tokens using mean pooling to get a single vector per entity per sentence
- Average across all sentences in a time period to get a temporal embedding

### 2. Change-Point Detection
- Compute cosine distance between consecutive temporal embeddings
- Apply detection algorithms: Threshold-based, PELT, Binary Segmentation 

### 3. Visualization
- Distance time series with detected events marked
- Self-similarity heatmap across all periods
- PCA projection showing semantic drift over time

## Installation

```bash
git clone https://github.com/altakachmohammad-del/diachronic-event-detection.git
cd diachronic-event-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python src/01_extract_embeddings.py
python src/02_change_point_detection.py
python src/03_visualize.py
```

## Models

| Model | Language | Use Case |
|-------|----------|----------|
| distilbert-base-multilingual-cased | Multilingual | Local development (Mac ARM compatible) |
| almanach/camembert-base | French | Production (GPU recommended) |
| flaubert/flaubert_base_cased | French | Alternative |

## Key References

1. Giulianelli et al. (2020) - Analysing Lexical Semantic Change with Contextualised Word Representations (ACL)
2. Hamilton et al. (2016) - Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change (ACL)
3. Beloued (2025) - Traitement automatique des evenements mediatiques (CORIA-TALN)
4. Kutuzov et al. (2018) - Diachronic Word Embeddings and Semantic Shifts: A Survey
5. Martin et al. (2020) - CamemBERT: a Tasty French Language Model (ACL)

## Roadmap

- [x] Baseline pipeline with simulated data
- [ ] Integration with INA corpus (real French news data)
- [ ] Multi-entity analysis (political figures, organizations)
- [ ] Advanced change-point detection methods
- [ ] CamemBERT fine-tuning experiments
- [ ] Evaluation against human-annotated events
- [ ] Final report and defense


## License

MIT License

## Acknowledgments

- **LIASD** (Laboratoire d'Informatique Avancee de Saint-Denis) - Universite Paris 8
- **INA** (Institut National de l'Audiovisuel)
- Supervisors: Aurélien Bossard (LIASD) & Abdelkrim Beloued (INA)
