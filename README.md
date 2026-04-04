# Skating Aerial Alignment

Python project to study the aerial phase of a figure-skating jump as a function
of the alignment between angular momentum and the body longitudinal axis.

## Development

The recommended local interpreter is:

```bash
/Users/mickaelbegon/miniconda3/envs/vitpose-ekf/bin/python
```

Run the tests with:

```bash
/Users/mickaelbegon/miniconda3/envs/vitpose-ekf/bin/python -m pytest -q
```

Launch the GUI with:

```bash
PYTHONPATH=src /Users/mickaelbegon/miniconda3/envs/vitpose-ekf/bin/python skating_aerial_alignment_gui.py
```

In the GUI:

- `Stabiliser le tronc` activates the current PD controller.
- `Auto PD` runs a short sub-optimal search to tune the PD gains for the
  current flight condition.
