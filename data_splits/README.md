# Data Split Manifests

The accepted-release package will include GUID-level manifests for the released splits:

- `train_guids.txt`
- `dev_guids.txt`
- `public_test_guids.txt`
- `protected_guids_blind.json`

Each `.txt` file contains one GUID per line. The JSON manifest records split metadata, counts, and blind-release status without redistributing third-party article text.

During the blind period, protected GUIDs may be redacted or sealed. After the blind period ends, the protected GUID list will be released under the same schema when redistribution constraints allow it.
