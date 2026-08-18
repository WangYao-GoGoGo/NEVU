# Data Split Manifests

The accepted-release package will include GUID-level manifests for the released splits:

- `train_guids.txt`
- `dev_guids.txt`
- `public_test_guids.txt`
- `protected_guids_blind.json`

Each `.txt` file contains one GUID per line. The JSON manifest records split metadata, counts, and blind-release status without redistributing third-party article text.

During the blind period, protected GUIDs may be redacted or sealed. The protected GUID list and labels will be released under the same schema in January 2027, when redistribution constraints allow it.

The Multi-Group Candidate Acceptance and Agreement Analysis datasets are also withheld until January 2027 because their type-balanced sampling includes blind instances. The Controlled Grouping Test against Heuristic Baselines materials are likewise withheld until January 2027 because they include blind materials.
