# Laboratoire de simulation

Ce dossier contient des exemples de scénarios batch pour lancer des campagnes
de simulations sans passer par la GUI.

## Fichiers

- `scenarios.json` : scénarios de démonstration prêts à lancer

## Lancer le batch d'exemple

```bash
cd /Users/mickaelbegon/Documents/Skating_jmp
PYTHONPATH=src /Users/mickaelbegon/miniconda3/envs/vitpose-ekf/bin/python -m skating_aerial_alignment batch \
  --config artifacts/lab/scenarios.json \
  --output-dir artifacts/lab/batch_demo
```

## Comparer les résultats

```bash
cd /Users/mickaelbegon/Documents/Skating_jmp
PYTHONPATH=src /Users/mickaelbegon/miniconda3/envs/vitpose-ekf/bin/python -m skating_aerial_alignment compare \
  --batch-dir artifacts/lab/batch_demo \
  --metric twist_turns
```

## Exporter les figures d'un run

```bash
cd /Users/mickaelbegon/Documents/Skating_jmp
PYTHONPATH=src /Users/mickaelbegon/miniconda3/envs/vitpose-ekf/bin/python -m skating_aerial_alignment export-plots \
  --run-dir artifacts/lab/batch_demo/run_001_baseline_backspin
```
