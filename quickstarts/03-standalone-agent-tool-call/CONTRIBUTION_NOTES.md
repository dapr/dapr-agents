# Bug Report & Proposed Fixes

This document summarizes the issues found in the `03-standalone-agent-tool-call` quickstart and the technical fixes implemented to enable a successful run on Windows and ensure YAML validation.

## 1. YAML Syntax Error (Indentation)
**Issue:** Several component YAML files had invalid indentation. The list items (`- name:`) under the `metadata:` block were aligned vertically with the `metadata` key itself.
**Impact:** Dapr failed to start with the error: `yaml: line 9: did not find expected key`.
**Fix:** Indented all list items under `metadata:` by 2 extra spaces (total 4 spaces from the start).

**Affected Files:**
- `components/openai.yaml`
- `components/historystore.yaml`
- `components/agentstatestore.yaml`
- `components/workflowstatestore.yaml`

## 2. Missing Windows (PowerShell) Support
**Issue:** The README only provided Bash commands (`export`, `grep`, `xargs`). These do not work in native Windows CMD or PowerShell.
**Impact:** Windows users encountered `'export' is not recognized` errors.
**Fix:** Added a dedicated "Windows (PowerShell)" section in the README with compatible commands for loading `.env` files and managing variables.

## 3. Undocumented Requirement: `dapr init`
**Issue:** The README did not mention that the Dapr runtime must be initialized before the first run.
**Impact:** Users received `fork/exec ... daprd.exe: The system cannot find the path specified`.
**Fix:** Added `dapr init` as a core prerequisite in the README.

## 4. Virtual Environment Isolation
**Issue:** Running `dapr run ... python script.py` often defaulted to the system Python, missing the `dapr-agents` library installed via `uv`.
**Impact:** `ModuleNotFoundError: No module named 'dapr_agents'`.
**Fix:** Updated all execution commands in the README to use `uv run`. This ensures the command executes within the correct virtual environment with all dependencies available.

## 5. Relative Path Cleanup
**Issue:** The README referenced `../../.env` but the `.env` file is commonly placed inside the quickstart folder itself for local testing.
**Fix:** Updated instructions to reference the local `.env` and added troubleshooting tips for path errors.
