# Provenance: Drasi unpacked event models

The files under `./dapr_agents/ext/drasi/schemas/unpacked/` are generated, do not edit them by hand.
They are derived from Drasi's `output-unpacked` TypeSpec contract (`Drasi.Unpacked`, v1).

Upstream source:        https://github.com/drasi-project/drasi-platform  
Contract path:          `typespec/output-unpacked/main.tsp`  
Vendored from commit:   `3edd930`  
Vendored on:            `2026-06-24`  

## How the vendored schema was produced
```bash
cd drasi-platform/typespec
npm install
npm run build ./output-unpacked
```
The generated `output-unpacked/_generated/@typespec/json-schema/*.yaml` JSON Schema files were then copied into `./dapr_agents/ext/drasi/schemas/output-unpacked/`.

## How the Python models are generated
```bash
./scripts/generate-drasi-models.sh
```