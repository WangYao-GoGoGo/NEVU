# Experimental Settings

This directory contains Linux shell commands for running the released experiment settings directly from the command line.

The original debug-style `sys.argv` snippets are preserved in `../experimental_settings_debug/`.

Each `.sh` file calls:

```bash
python unified_value_recognition.py ...
```

Run a setting from the repository root, for example:

```bash
bash experimental_settings/G4/G4_tfidf_config.sh
```

Dataset paths point to the public dataset layout under `dataset/event_base/` and `dataset/sub/`, using the original non-`_reformatted` files.
