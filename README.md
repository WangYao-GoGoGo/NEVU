# Event-Centric Human Value Understanding in News-Domain Texts

This repository contains the reproducibility materials for:

**Event-Centric Human Value Understanding in News-Domain Texts: An Actor-Conditioned, Multi-Granularity Benchmark**

The repository separates materials that reproduce the reported model evaluations from materials that document the benchmark construction process.

## Release Scope

We release the following materials for reproducibility:

- Inference prompts used for the reported model evaluations: `prompt/canonical_system.txt` and `prompt/canonical_prompt.txt`.
- Evaluation scripts: `evaluate.py`, `evaluate_deep_analysis2.py`, and the evaluation path in `unified_value_recognition.py`.
- Baseline and fine-tuning entry points: `unified_value_recognition.py`, `inference.py`, and `nonllm_retrieval.py`.
- Baseline and fine-tuning settings: `experimental_settings/`.
- Human-value taxonomy and label mappings: `dataset/hv/`.
- Split manifest schema and release notes: `data_splits/`.
- Benchmark construction prompts and reference scripts: `benchmark_construction_related_resources/`.

The construction resources are released for transparency and auditability. They document the prompts, filtering scripts, and annotation-interface templates used during dataset creation, but they are not intended to exactly reconstruct the original collection pipeline because the pipeline depended on third-party news sources, APIs, and blind-test materials that cannot all be redistributed.

## Dataset

This repository provides code, prompts, settings, and reproducibility/reference materials only. The dataset files are released separately at:

https://anonymous.4open.science/r/nevu_repo-5D42/

## Reproducible Evaluation Materials

Use `experimental_settings/` as the source for the reported experimental settings. These files record the command-line arguments used for each model group and baseline.

Current release configs include:

- `experimental_settings/G1/`: API-based LLM inference settings.
- `experimental_settings/G2/`: open-source instruct-model inference settings.
- `experimental_settings/G3/`: LoRA fine-tuning, LoRA inference, and CGT settings.
- `experimental_settings/G4/`: TF-IDF and SBERT retrieval baseline settings.

The LoRA implementation uses rank 16, alpha 32, dropout 0.05, no bias terms, causal-language-model PEFT, and target modules `q_proj`, `k_proj`, `v_proj`, and `o_proj`. The G4 TF-IDF and SBERT examples include the retrieval voting, top-k, filtering, and conflict-resolution parameters used by `nonllm_retrieval.py`.

## Data Splits

The accepted-release package will include split manifests for:

- `train`
- `dev`
- `public_test`
- protected blind-test GUIDs after the blind period ends in January 2027

The split files will contain GUID-level membership only, not third-party article text that cannot be redistributed. See `data_splits/README.md` and `data_splits/split_manifest.schema.json` for the intended manifest format.

Some auxiliary analysis materials are also withheld during the blind period. The datasets used for the paper's Multi-Group Candidate Acceptance and Agreement Analysis include blind instances because the sampling procedure preserved type balance. These analysis datasets will be released in January 2027. The Controlled Grouping Test against Heuristic Baselines also uses blind materials and will be released in January 2027.

## Materials Not Redistributed

Some materials cannot be released in full:

- Third-party news article text or API responses when redistribution is restricted by provider terms.
- Credentials, API keys, private service endpoints, and local machine paths.
- Blind-test labels or protected GUID mappings before the blind evaluation period ends in January 2027.
- Multi-Group Candidate Acceptance and Agreement Analysis datasets before January 2027, because they include blind instances sampled to preserve type balance.
- Controlled Grouping Test against Heuristic Baselines materials before January 2027, because they also include blind materials.
- Full proprietary model weights or third-party model files governed by their original licenses.

Where full redistribution is not permitted, we provide GUID manifests, processing descriptions, prompt templates, and runnable evaluation scripts so that released dataset files can be evaluated consistently.

## Basic Usage

Install dependencies in a fresh Python environment:

```bash
pip install -r requirements.txt
```

Run a released configuration by following the corresponding argument list in `experimental_settings/`. For example, the G4 TF-IDF baseline corresponds to:

```bash
python unified_value_recognition.py \
  --mode infer \
  --group G4 \
  --model_name g4-tfidf-retrieval \
  --label_space L1 \
  --train_event dataset/training_dataset_total_formatted.json \
  --train_hv dataset/train_sub_gold.json \
  --dev_event dataset/dev_dataset_total_formatted.json \
  --dev_hv dataset/dev_sub_gold.json \
  --test_event dataset/test_dataset_total_formatted.json \
  --test_hv dataset/test_sub_gold.json \
  --canonical_prompt prompt/canonical_prompt.txt \
  --canonical_system prompt/canonical_system.txt \
  --hv1_label_dir dataset/hv/id2hv.json \
  --hv_label_dir dataset/hv/values.json \
  --l1tol2_mappings dataset/hv/l1tol2_id_mapping.json \
  --retrieval_method tfidf \
  --retrieval_k 5 \
  --retrieval_vote_threshold 0.3 \
  --retrieval_min_sim 0.0 \
  --retrieval_level_filter true \
  --retrieval_conflict_mode prefer_higher \
  --tfidf_ngram_max 2 \
  --tfidf_min_df 2 \
  --tfidf_max_df 0.95 \
  --tfidf_max_features 200000
```
