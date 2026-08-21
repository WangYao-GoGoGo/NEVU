# Repository Contents

This file maps the main files and directories in the release. The README explains the overall release policy; this file is a compact guide to what each part is for.

## Core Evaluation Code

- `unified_value_recognition.py`: main experimental runner for the reported G1-G4 settings.
- `inference.py`: inference utilities used by the main runner.
- `nonllm_retrieval.py`: TF-IDF and SBERT retrieval baselines used for G4.
- `evaluate.py`: evaluation script for polarity-aware multilabel human-value recognition.
- `utils/`: shared helpers for data loading, prompts, model/API calls, file handling, and experiment utilities.
- `config/`: configuration loader code and example construction/evaluation configuration files.

## Prompts

- `prompt/canonical_system.txt`: system/developer message used by the reported inference pipeline.
- `prompt/canonical_prompt.txt`: canonical inference prompt template used by the reported model-evaluation pipeline.

## Experimental Settings

`experimental_settings/` records Linux shell commands for the reported groups. Each `.sh` file calls `python unified_value_recognition.py ...` with explicit command-line arguments:

- `G1/`: API-based LLM inference settings.
- `G2/`: open-source instruct-model inference settings.
- `G3/`: LoRA fine-tuning settings.
- `G4/`: TF-IDF and SBERT retrieval baseline settings.

`experimental_settings_debug/` preserves the original debug-style `sys.argv` snippets that were used during development.

The LoRA settings are implemented in `unified_value_recognition.py` with rank 16, alpha 32, dropout 0.05, no bias terms, causal-language-model PEFT, and target modules `q_proj`, `k_proj`, `v_proj`, and `o_proj`.

The G4 command files correspond to `nonllm_retrieval.py` and include TF-IDF/SBERT retrieval options, top-k selection, similarity thresholds, vote thresholds, level filtering, and conflict-resolution parameters.

## Dataset Reference Files

- `dataset/hv/`: human-value taxonomy, label mappings, level mappings, and direction mappings.
- `dataset/news/`: news type taxonomy/reference files.
- `dataset/wiki_ontology/`: ontology reference files used by construction and preprocessing utilities.

The actual benchmark dataset files are released separately at:

https://anonymous.4open.science/r/nevu_repo-5D42/

## Benchmark Construction Resources

`benchmark_construction_related_resources/` documents the construction workflow. These materials are released for transparency and auditability, not as a complete rerunnable data-collection pipeline.

### Phase 1

- `Phase-1/stage-3/prompt_news_type_classification.txt`: news type classification prompt using the ten-category news type taxonomy.

### Phase 2

- `Phase-2/prompt_actor_mapping.txt`: actor mapping prompt.
- `Phase-2/prompt_article_segment_with_heading.txt`: article segmentation prompt.
- `Phase-2/prompt_article_type_classification.txt`: article/news type classification prompt used in the construction workflow.
- `Phase-2/prompt_event_extraction.txt`: event-centric extraction prompt.
- `Phase-2/prompt_subevent_mapping.txt`: subevent mapping prompt.

### Phase 3

- `Phase-3/stage1/prompt/`: open human-value recognition prompts for article, subevent, behavior-based composite event, and story-based composite event levels.
- `Phase-3/stage2/prompt/qa_based_verification/`: QA-based verification prompts for article, subevent, behavior-based composite event, and story-based composite event labels.
- `Phase-3/stage3/label_studio/`: Label Studio interface templates for human annotation and verification.
- `Phase-3/stage4/prompt/`: prompts for verifying behavior-based and story-based composite-event labels using human-reviewed subevent-level results.

## Release Limitations

This repository does not release private credentials, local paths, third-party article text or API responses whose provider terms prohibit redistribution, full third-party model weights, protected blind-test labels before January 2027, Multi-Group Candidate Acceptance and Agreement Analysis datasets before January 2027, or Controlled Grouping Test materials before January 2027.

Local development artifacts such as `.idea/`, `.DS_Store`, `__pycache__/`, and `*.pyc` are not part of the intended release package.
