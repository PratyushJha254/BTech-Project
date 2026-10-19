# Machine Learning and Monte Carlo Simulation of Ising-Model Phase Transitions

B.Tech project, Department of Physics, IIT Roorkee (2023–24), by Pratyush Jha and Sudatt Kayal, under Prof. Peayush Choubey.

📄 [Pratyush's thesis](https://drive.google.com/drive/folders/1DKc8DTECObGSjF4NVKis8U_D-Kcc0_Kx) · 📄 [Presentation](BTP%20Presentation%20Pratyush%20and%20Sudatt.pdf)

## Motivation

Frustrated Josephson-junction arrays map to long-range Ising models. Monte Carlo simulation of these models is expensive, because every spin interacts with every other spin. As a first step toward reducing the Monte Carlo data such systems need, this project asks how well a neural-network phase classifier trained on one Ising model transfers to another. It moves from the nearest-neighbour (NN) model to the next-nearest-neighbour (NNN) model

H = J_nn Σ_⟨ij⟩ s_i s_j + J_nnn Σ_⟨⟨ik⟩⟩ s_i s_k,  with coupling ratio R = J_nnn / J_nn.

For R < 0.5 the ordered phase is antiferromagnetic; for R > 0.5 it is super-antiferromagnetic.

## Methods

- **Data:** Metropolis Monte Carlo sampling of NN (ferromagnetic and antiferromagnetic) and NNN Ising configurations on square lattices.
- **Classifiers (PyTorch):**
  - A fully connected network with the architecture of Carrasquilla & Melko, [*Nature Physics* 13, 431 (2017)](https://doi.org/10.1038/nphys4035).
  - A CNN with 64 2×2 filters.
  - A deeper CNN (CNN2: three 3×3 convolutional layers with max pooling).
- **Zero-shot transfer:**
  - Classifiers trained on one model are tested on another without retraining.
  - Ferromagnetic and antiferromagnetic data are related by the transformation s_ij → (−1)^(i+j) s_ij.
- **Fine-tuning:**
  - The convolutional feature extractor of a pre-trained CNN is frozen and the fully connected classifier is retrained on a small dataset.
  - The classifier layers (FC1, FC2) are either initialized from the pre-trained weights or re-initialized, and the variants are compared.

## Results

**Phase classification on the NN Ising model** (test accuracy):

| Transition | Model | L = 10 | L = 20 | L = 30 |
|---|---|---|---|---|
| Ferromagnetic → paramagnetic | FCNN | 93.7% | 96.6% | 97.2% |
| | CNN | 97.9% | 98.8% | 99.5% |
| Antiferromagnetic → paramagnetic | FCNN | 87.0% | 91.0% | 94.7% |
| | CNN | 97.5% | 98.9% | 99.2% |

**Ferromagnetic ↔ antiferromagnetic (zero-shot):** using the (−1)^(i+j) transformation, models trained on one case classify the other with about 96% accuracy (FCNN and CNN, both directions).

**NN → NNN (zero-shot):** models trained on the NN antiferromagnet, tested on NNN data (L = 64):

| Model | R = −1 | R = 0.45 | R = +2 |
|---|---|---|---|
| CNN | 99.2% | 96.0% | 58.0% |
| FCNN | 89.0% | 80.8% | 51.2% |

The models transfer well while the ordered phase stays antiferromagnetic (R < 0.5). They do not transfer at R = +2, where the ordered phase is super-antiferromagnetic.

**Fine-tuning:**
- *NN case.* Fine-tuning on 256 configurations reaches 98% accuracy between the ferromagnetic and antiferromagnetic cases. This holds when the classifier is re-initialized; with both FC layers kept at their pre-trained weights, accuracy is about 50%.
- *R = +2 case.* CNN2 trained directly reaches 95.5%. Fine-tuning a CNN2 pre-trained on the NN ferromagnet reaches 86.4%.

## Notebooks

**Monte Carlo and NN Ising classification**
- `IsingModel_Metropolis.ipynb`: Metropolis sampling of NN Ising configurations.
- `NNs_trainedOnFerro_testedOnAntiFerro_NNIsingModel.ipynb`: FCNN and CNN on the NN model; ferromagnetic ↔ antiferromagnetic transfer.

**NN → NNN transfer**
- `NNs_trainedOnFerroNN_testedOnNNN_IsingModel.ipynb`: NNN data generation and zero-shot tests.
- `Model_interpretations.ipynb`: CNN feature maps and tests on NNN data for several values of R.
- `TransferLearning_in_NNIsing.ipynb`: fine-tuning between the ferromagnetic and antiferromagnetic NN cases, comparing classifier initializations.
- `TransferLearning_in_NNN_plus2.ipynb`, `IInd_TransferLearning_in_NNN_plus2.ipynb`: CNN2 and fine-tuning for R = +2.
- `TSNE.ipynb`: t-SNE embeddings of ferromagnetic, antiferromagnetic and super-antiferromagnetic configurations.

**Extensions**
- `LRIM_TransferLearning_sigma0_8.ipynb`: fine-tuning on long-range Ising configurations (σ = 0.8).
- `lattice_ani_Binder_cumu.ipynb`: Binder cumulant and thermal analysis of T_c for the long-range model; lattice animations.
- `TriangularIsingModel.ipynb`: Metropolis simulation of the triangular-lattice Ising model.
- `XYModel_MetropolisAlgorithm.ipynb`, `XYModel_MetropolisAlgorithm_2.ipynb`, `XYModel_CNN.ipynb`: Metropolis simulation of the 2D XY model (including the helicity modulus) and CNN classification of XY configurations.

## Running

The notebooks were written for Google Colab (badge at the top of each). The Monte Carlo datasets (`.npz`) are not included because of their size; they can be regenerated with the Metropolis code in the notebooks.
