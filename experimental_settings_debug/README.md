# Debug Experimental Settings

This directory preserves the original debug-style experiment settings used during development.

Each file contains a `sys.argv = [sys.argv[0]] + [...]` argument block. These snippets are intended for running or debugging experiments directly inside Python or VS Code.

To use one setting:

1. Open the corresponding file under `experimental_settings_debug/G1` to `G4`.
2. Copy the `sys.argv` block.
3. Replace the `DEBUG` argument-setting block in the target Python program with that block.
4. Run the Python program from the IDE or debugger.

For command-line execution on Linux, use the converted shell commands in `experimental_settings/` instead.
