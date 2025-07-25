# 2D Multi-Modal Data Fusion for Electron Microscopy

Fused multi-modal electron microscopy, a combines elastic scattering (HAADF) and inelastic spectroscopic signals (EELS/EDX) to recover high signal-to-noise ratio chemical maps at nano- and atomic-resolution. 


By linking simultaneously acquired modalities through regularized optimization, the method can reduce dose requirements by over one order of magnitude while substantially improving SNR for chemical maps. 

## Installation 

The package is available on PyPI:
```bash
pip install multimodal-fusion
```
For local development, clone the repository and install in editable mode:
```bash
git clone https://github.com/jtschwar/Multi-Modal-2D-Data-Fusion.git --recursive
cd Multi-Modal-2D-Data-Fusion
pip install -e . 
```

## Quick Start

```python
from multimodal_fusion import DataFusion

# Initialize fusion with list of elements
elements = ['Co', 'S', 'O']
fusion = DataFusion(elements)

# Load your chemical maps
# Provide as a dictionary where keys match your element list
# This helper function loads starter data from github
cobalt_map, sulfur_map, oxygen_map, haadf_im = fusion.load_edx_example()
chemical_maps = {
    'Co': cobalt_map,      # 2D numpy arrays
    'S': sulfur_map, 
    'O': oxygen_map
}
fusion.load_chemical_maps(chemical_maps)

# Load the simultaneously acquired HAADF image
fusion.load_haadf(haadf_im)  # 2D numpy array

# Run the fusion algorithm 
# We can adjust with regularization parameters
fusion.run(
    nIter=50, 
    lambdaEDS = 0.005, lambdaTV=0.1,
    plot_images=True, plot_convergence=True)

# Get results in dictionary format
results = fusion.get_results()
fused_cobalt = results['Co']
fused_sulfur = results['S']
```
#### Documentation
A comprehensive tutorial for learning how to adjust the hyperparameters is available: [J. Manassa, M. Shah, et. al. "Fused Multi-Modal Electron Microscopy - A Beginner's Guide, _Elemental Microscopy_ (2024).](https://www.elementalmicroscopy.com/articles/EM000003)

## Citation

If you use any of the data and source codes in your publications and/or presentations, we request that you cite our papers:

[J. Schwartz, Z.W. Di, et. al., "Imaging atomic-scale chemistry from fused multi-modal electron microscopy", _npj Comput. Mater._ **8**, 16 (2022).](https://www.nature.com/articles/s41524-021-00692-5)


A tutorial for learning how to adjust the hyper-parameters is also available here: [J. Manassa, M. Shah, et. al. "Fused Multi-Modal Electron Microscopy - A Beginner's Guide, _Elemental Microscopy_ (2024).](https://www.elementalmicroscopy.com/articles/EM000003)


