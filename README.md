# 2D Multi-Modal Data Fusion for Electron Microscopy

Fused multi-modal electron microscopy, a combines elastic scattering (HAADF) and inelastic spectroscopic signals (EELS/EDX) to recover high signal-to-noise ratio chemical maps at nano- and atomic-resolution. 


By linking simultaneously acquired modalities through regularized optimization, the method can reduce dose requirements by over one order of magnitude while substantially improving SNR for chemical maps. 

## Installation 

```bash
pip install multimodal-fusion
```

## Quick Start

```python
# Initialize fusion with list of elements
elements = ['Co', 'S', 'O']
fusion = DataFusion(elements)

# Load your chemical maps (from any software - ImageJ, Digital Micrograph, etc.)
# Provide as a dictionary where keys match your element list
chemical_maps = {
    'Co': cobalt_map,      # 2D numpy arrays
    'S': sulfur_map, 
    'O': oxygen_map
}
fusion.load_chemical_maps(chemical_maps)

# Load the simultaneously acquired HAADF image
fusion.load_haadf(haadf_image)  # 2D numpy array

# Run the fusion algorithm 
# We can adjust with regularization parameters
fusion.run(nIter=50, lambdaTV=0.1)

# Get results in dictionary format
results = fusion.get_results()
fused_cobalt = results['Co']
fused_sulfur = results['S']
```

## Citation

If you use any of the data and source codes in your publications and/or presentations, we request that you cite our papers:

[J. Schwartz, Z.W. Di, et. al., "Imaging atomic-scale chemistry from fused multi-modal electron microscopy", _npj Comput. Mater._ **8**, 16 (2022).](https://www.nature.com/articles/s41524-021-00692-5)


A tutorial for learning how to adjust the hyper-parameters is also available here: [J. Manassa, M. Shah, et. al. "Fused Multi-Modal Electron Microscopy - A Beginner's Guide, _Elemental Microscopy_ (2024).](https://www.elementalmicroscopy.com/articles/EM000003)

