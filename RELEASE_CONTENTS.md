# Repository Contents

This file summarizes the role of the main directories and scripts in the release. The README provides the general release policy; this file is a compact map of what each part of the repository is for.

## Benchmark Construction Resources

`benchmark_construction_related_resources/` contains reference materials for dataset construction. These files document the prompts, scripts, and annotation-interface templates used during benchmark creation.

The directory is organized by construction phase:

- `Phase-1/`: initial news collection, topic/type filtering, balancing utilities, and sensitivity checking resources.
- `Phase-2/`: event-centric article processing prompts, including actor mapping, article segmentation, event extraction, subevent mapping, and article-type classification.
- `Phase-3/`: human-value recognition and verification resources, including construction prompts, QA-based verification prompts, Label Studio interface templates, and human-check verification prompts.

These construction resources are released for transparency and auditability. They are not intended to fully reproduce the original data collection process because some steps depended on third-party news sources, external APIs, human annotation, and blind-test materials.

## Experimental Code

The files outside `benchmark_construction_related_resources/` are primarily for model inference, fine-tuning, evaluation, and analysis.

- `unified_value_recognition.py`: main experimental runner for G1-G4 settings.
- `inference.py`: inference utilities used by the main runner.
- `nonllm_retrieval.py`: TF-IDF and SBERT retrieval baselines for G4.
- `evaluate.py`: polarity-aware multilabel evaluation.
- `evaluate_deep_analysis2.py`: extended analysis utilities.

Additional experiment variants:

- `unified_value_recognition_ablation_study.py`: ablation-study runner.
- `unified_value_recognition_error_analysis.py`: error-analysis runner.
- `unified_value_recognition_cgt.py`: Controlled Grouping Test against heuristic baselines.
- `inference_cgt.py`: inference utilities for the Controlled Grouping Test.

## Experimental Settings

`experimental_settings/` contains the argument settings used for the reported experimental groups:

- `G1/`: API-based LLM inference settings.
- `G2/`: open-source instruct-model inference settings.
- `G3/`: LoRA fine-tuning, LoRA inference, and CGT settings.
- `G4/`: TF-IDF and SBERT retrieval baseline settings.
- `error_analysis/`: settings for error-analysis runs.

The LoRA implementation in `unified_value_recognition.py` uses rank 16, alpha 32, dropout 0.05, no bias terms, causal-language-model PEFT, and target modules `q_proj`, `k_proj`, `v_proj`, and `o_proj`.

## Prompts And Splits

- `prompt/`: inference prompt and system message used by the reported model-evaluation pipeline.
- `data_splits/`: split-manifest schema and protected blind-split template.
- `dataset/hv/`: human-value taxonomy and label mappings.
- `dataset/news/` and `dataset/wiki_ontology/`: news-type and ontology reference files.

## Non-Released Materials

We do not release private credentials, local paths, third-party article text or API responses whose provider terms prohibit redistribution, full third-party model weights, or blind-test labels before the blind period ends.
