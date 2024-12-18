# AI Design Assistant Source Code
---
This code contains an AI-based design assistant suitable for the [Axial Turbine](https://github.com/ihs-ustutt/axial_turbine_database/tree/main).
Computational Fluid Dynamic (CFD) simulation data is stored in tensors for clustering turbines based on flow field similarities.
Clustering results are mapped by a multi-class prediction neural network to estimated flow field properties for new turbines without CFD.
For a more detailed explanation see: 

Eyselein,  S.; Tismer,  A.; Raj,  R.; Rentschler,  T.; Riedelbauch,  S. AI-Based Clustering of Numerical Flow Fields for Accelerating the Optimization of an Axial Turbine. Preprints 2024, 2024121380. https://doi.org/10.20944/preprints202412.1380.v1

See example in run_ai_design_assistant.ipynb.

---
### Dataset Availability

Flow field tensors are available on request. (runData + tensors)

---
### Requirements

Before running this code install and source [dtOO](https://github.com/ihs-ustutt/dtOO).
It is recommended to run this code on a cuda-capable device. If not, pretrained model data won't 
work. Delete files in /model_trained and retrain in this case.

For Python environment see requirements.txt.

---
### License

This work is licensed under the [MIT license](/LICENSE). 