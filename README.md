# 2D Multi-Modal Data Fusion for Electron Microscopy

Fused multi-modal electron microscopy, a combines elastic scattering (HAADF) and inelastic spectroscopic signals (EELS/EDX) to recover high signal-to-noise ratio chemical maps at nano- and atomic-resolution. 


By linking simultaneously acquired modalities through regularized optimization, the method can reduce dose requirements by over one order of magnitude while substantially improving SNR for chemical maps. 

## Installation 

```bash
pip install multimodal-fusion
```

## Quick Start

```python
from multimodal_fusion import DataFusion

# Load Your Data
fusion = DataFusion()

data = fusion.run()
```

## Citation

If you use any of the data and source codes in your publications and/or presentations, we request that you cite our papers:

[J. Schwartz, Z.W. Di, et. al., "Imaging atomic-scale chemistry from fused multi-modal electron microscopy", _npj Comput. Mater._ **8**, 16 (2022).](https://www.nature.com/articles/s41524-021-00692-5)


A tutorial for learning how to adjust the hyper-parameters is also available here: [J. Manassa, M. Shah, et. al. "Fused Multi-Modal Electron Microscopy - A Beginner's Guide, _Elemental Microscopy_ (2024).](https://www.elementalmicroscopy.com/articles/EM000003)

