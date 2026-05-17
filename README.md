# helios-fusion-engine

[![CI](https://github.com/577Industries/helios-fusion-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/577Industries/helios-fusion-engine/actions/workflows/ci.yml) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/helios-fusion-engine.svg)](https://pypi.org/project/helios-fusion-engine/)

> Model-agnostic probabilistic fusion of heterogeneous space-weather model outputs: Bayesian Model Averaging orchestrator, isotonic-regression reliability calibrator, split + Mondrian conformal prediction wrappers, and severity-stratified validation harness with CCMC-compatible metrics (HSS, TSS, POD, FAR, Brier, CRPS). Framework only; trained weights/configs live in the private helios-fusion-internal companion repo.

## Status

This repository is part of the **HELIOS** program — a NASA SBIR Phase I effort by
577 Industries Inc. supporting subtopic SPWX.1.S26A (Advanced Data-Driven
Applications for Space Weather R2O2R). See proposal §2 Obj. 2 + §3.1 (pre-registered validation) + §4.2 innovation #1 of the proposal.

**Initial scaffolding committed 2026-05-17. Implementation in progress.**
Open issues to comment on the design or propose contributions.

## Quickstart

```bash
pip install helios-fusion-engine
```

```python
import helios_fusion
print(helios_fusion.__version__)
```

## Documentation

- **Master plan**: see [`helios-program`](https://github.com/577Industries/helios-program) (private; internal team)
- **Specification**: docs published at the project's docs site when available
- **Provenance**: every output traces to its upstream model and transformation chain
  via [`helios-provenance-spec`](https://github.com/577Industries/helios-provenance-spec)

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Substantive changes should be discussed in an issue first.

## Citation

```bibtex
@software{helios_helios_fusion_engine,
  author       = {Waweru, Thomas and 577 Industries Inc.},
  title        = { helios-fusion-engine: Model-agnostic probabilistic fusion of heterogeneous space-weather model outputs: Bayesian Model Averaging orchestrator, isotonic-regression reliability calibrator, split + Mondrian conformal prediction wrappers, and severity-stratified validation harness with CCMC-compatible metrics (HSS, TSS, POD, FAR, Brier, CRPS) },
  year         = {2026},
  publisher    = {577 Industries Inc.},
  url          = {https://github.com/577Industries/helios-fusion-engine},
}
```
