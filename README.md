# Event-Centric Human Value Understanding in News-Domain Texts

This repository contains reproducibility and release materials for:

**Event-Centric Human Value Understanding in News-Domain Texts: An Actor-Conditioned Benchmark across Multi-Scope Event Contexts**

The repository is organized around two goals:

1. Reproduce the reported model-evaluation pipeline with released dataset files.
2. Document the benchmark-construction prompts and annotation resources used to build the benchmark.

## Dataset

This repository provides code, prompts, settings, and reference materials only. The dataset files are released separately at:

https://anonymous.4open.science/r/nevu_repo-5D42/

The dataset release is expected to provide the released train/dev/public-test files and their corresponding identifiers. Third-party news article text or API responses are not redistributed here when redistribution is restricted by provider terms.

## Release Scope

This repository includes the following release materials:

- Inference prompts for the reported evaluation pipeline: `prompt/canonical_system.txt` and `prompt/canonical_prompt.txt`.
- Main experimental runner and inference code: `unified_value_recognition.py` and `inference.py`.
- Evaluation code: `evaluate.py`.
- Non-LLM retrieval baselines: `nonllm_retrieval.py`.
- Experimental settings for G1-G4: `experimental_settings/`.
- Human-value taxonomy and label mappings: `dataset/hv/`.
- News-type and ontology reference files: `dataset/news/` and `dataset/wiki_ontology/`.
- Benchmark-construction prompts and annotation resources: `benchmark_construction_related_resources/`.
- Dependency list: `requirements.txt`.

The construction resources are provided for transparency and auditability. They document prompts, type classification, event processing, human-value verification, and annotation-interface templates. They are not intended to fully reconstruct the original collection pipeline because some steps depended on third-party sources, APIs, human annotation, and blind-test materials that cannot be redistributed now.

## Experimental Settings

Use `experimental_settings/` as the source for the reported baseline and fine-tuning settings:

- `experimental_settings/G1/`: API-based LLM inference settings.
- `experimental_settings/G2/`: open-source instruct-model inference settings.
- `experimental_settings/G3/`: LoRA fine-tuning settings.
- `experimental_settings/G4/`: TF-IDF and SBERT retrieval baseline settings.

The LoRA implementation in `unified_value_recognition.py` uses rank 16, alpha 32, dropout 0.05, no bias terms, causal-language-model PEFT, and target modules `q_proj`, `k_proj`, `v_proj`, and `o_proj`.

The TF-IDF and SBERT baseline settings are recorded in `experimental_settings/G4/` and implemented in `nonllm_retrieval.py`. These settings include retrieval method, top-k selection, similarity filtering, vote thresholding, level filtering, conflict handling, and TF-IDF/SBERT model parameters.

## Benchmark Construction Resources

`benchmark_construction_related_resources/` contains reference materials for the construction process:

- `Phase-1/`: news type classification, possible human-value recognition resources, and keyword resources used during early filtering and collection.
- `Phase-2/`: event-centric article processing prompts, including actor mapping, article segmentation, news/article type classification, event extraction, and subevent mapping.
- `Phase-3/`: human-value recognition and verification prompts, QA-based verification prompts, Label Studio annotation templates, and verification prompts based on human-reviewed results.

The terminology in these prompts follows the paper's naming for composite events:

- `behavior-based composite event`
- `story-based composite event`

Some files under construction resources are reference-only materials from the original construction workflow. They are useful for auditing the process, but they do not by themselves reproduce the full benchmark because source articles, API responses, and blind materials are not fully redistributable.

## Blind And Withheld Materials

Some materials are withheld during the blind period:

- Protected blind-test GUIDs and labels will be released after the blind period ends in January 2027.
- The paper's Multi-Group Candidate Acceptance and Agreement Analysis datasets include blind instances because the sampling procedure preserved type balance. These analysis datasets will be released in January 2027.
- Controlled Grouping Test against Heuristic Baselines materials also include blind materials and will be released in January 2027.

Until then, the release provides public evaluation code, prompt templates, settings, and dataset links for the non-blind release materials.

## Materials Not Redistributed

The following materials are not redistributed in this repository:

- Third-party news article text or API responses when redistribution is restricted.
- Private credentials, API keys, private endpoints, and local machine paths.
- Protected blind-test labels and protected GUID mappings before January 2027.
- Multi-Group Candidate Acceptance and Agreement Analysis datasets before January 2027.
- Controlled Grouping Test materials before January 2027.
- Full proprietary model weights or third-party model files governed by their original licenses.
- Local development files such as IDE metadata, cache files, and Python bytecode.

## Basic Usage

Install dependencies in a fresh Python environment:

```bash
pip install -r requirements.txt
```

Run experiments by following the argument lists in `experimental_settings/`. For example, the G4 TF-IDF retrieval baseline corresponds to the settings in:

```text
experimental_settings/G4/G4_tfidf_config.json
```

The released dataset files should be placed according to the paths expected by the selected configuration, or the paths in the configuration should be adjusted to the local dataset location.
