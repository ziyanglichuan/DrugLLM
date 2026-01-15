# DrugLLM

Large Language Models (LLMs) have revolutionized machine learning with their few-shot learning and reasoning capabilities, demonstrating impressive results in fields like natural language processing and computer vision. However, when applied to the domains of biology and chemistry, current LLMs face substantial limitations, particularly in capturing the nuanced relationships between molecular structure and pharmacochemical properties. This challenge has constrained the application of few shot learning for small-molecule generation and optimization in drug discovery. Here, we introduce DrugLLM, a novel LLM tailored specifically for molecular optimization. DrugLLM leverages Functional Group Tokenization (FGT), which effectively tokenizes molecules for LLM learning, achieving over 53% token compression compared to SMILES. Besides, we propose a new pre-training strategy that enables DrugLLM to iteratively predict and modify molecular structures based on a few prior modifications, aligning each modification toward optimizing a specified pharmacological property.
In multiple computational experiments, DrugLLM achieved state-of-the-art performance in few-shot molecular generation, surpassing all the mainstream LLMs including GPT-4. Furthermore, by applying DrugLLM to optimize HCN2 inhibitors, two bioactive compounds were obtained and successfully validated through wet lab experiments. These results highlight the robust potential of DrugLLM in accelerating the optimization of molecules and AI-driven drug discovery.

For more information, please refer to our paper on [arXiv:2405.06690](https://arxiv.org/abs/2405.06690).

## Reproducibility & Quick Access (for Reviewers)

To facilitate reproducibility and ease of use, we provide:

- **User-friendly inference script**:  
  `simple_inference.py` (see **Simple Inference (Quick Start)**)

- **Training setup and hyperparameters**:  
  Detailed in **Chemprop Training Pipeline**

- **Evaluation and success rate testing**:  
  See **Property Generation and Success Rate Testing**

All scripts can be executed directly following the instructions below.


## Usage Guide

### Model Download
The DrugLLM model is hosted on Hugging Face and can be downloaded from [here](https://huggingface.co/ziyanglichuan/DrugLLM). After downloading, place the model files in the `model` directory.

### Training Dataset
The training dataset and supporting experimental data are available at [https://osf.io/f6yqn/](https://osf.io/f6yqn/).

### Testing Dataset
A sample test dataset is included in the `data_example` directory.

### Installation
To install the required dependencies, execute the following command:

```bash
pip install -r requirements.txt
```

> **Note**: This project is best supported on Python 3.9.


## Simple Inference (Quick Start)

We provide a minimal inference script `simple_inference.py` for quickly running DrugLLM on a small JSON input file and printing the results directly to the console.  
This script is intended for **quick validation, debugging, and demonstration purposes**, rather than large-scale evaluation.

### Features

- Automatically converts **SMILES → FGT** inside the instruction
- Supports **few-shot molecular modification** prompts
- Runs deterministic DrugLLM inference
- Extracts the generated **FGT** and decodes it back to **SMILES**

### Input JSON Format

The input file should be a JSON list. Each item follows the structure below:

```json
{
  "instruction": "...",
  "output": ""
}
````

#### `instruction` (Required)

The `instruction` field is a single string that may contain:

1. A **property optimization directive**, e.g.

   ```
   decrease the water solubility.
   ```

2. Multiple **few-shot molecular modification examples**, formatted as:

   ```
   {SMILES_before}->{SMILES_after}
   ```

   and concatenated using `.` as separators.

3. A final single {SMILES}, which represents the source molecule to be optimized.

**Example**:

```text
decrease the water solubility. Output: {CN(CCc1ccncc1)C(=O)NCCc1ccc(F)cc1}->{CN(CCc1ccncc1)C(=O)NCCc1ccc(Cl)cc1}.{Cc1nc2nnc(C(=O)Nc3ccc(Cl)cc3)c(C)n2n1}->{Cc1nc2nnc(C(=O)Nc3cccc(C(F)(F)F)c3)c(C)n2n1}.{O=S(=O)(c1cccc(F)c1)N1CCOc2ccc(F)cc21}->{Cc1cccc(S(=O)(=O)N2CCOc3ccc(Cl)cc32)c1}.{Cc1nnc(SCC(=O)Nc2nccs2)o1}
```

Each SMILES string enclosed in `{}` will be:

* Automatically converted to **FGT** before inference
* Used as few-shot context for molecular optimization

The **last `{SMILES}` (or `{FGT}` after conversion)** in the instruction is treated as the **source molecule** to be optimized.

#### `output` (Optional)

The `output` field is **not used** by `simple_inference.py` and can be left empty.
It is kept only for compatibility with training or evaluation datasets.

### Usage

Run the following command:

```bash
python simple_inference.py --input ../data_example/simple_test.json
```

---

### Data Preprocessing

The `data_example` directory provides sample input files that can be directly processed using the SMILES-to-FGT conversion script. To convert molecular data from SMILES format to Functional Group Tokenization (FGT), run:

```bash
python jsonSmiles_to_FGT.py <input_directory> <output_directory>
```

* **Parameters**:

  * `<input_directory>`: Path to the directory containing SMILES data files (e.g., files in `data_example`).
  * `<output_directory>`: Path where the converted FGT data will be saved.

**Example**:

```bash
python jsonSmiles_to_FGT.py ./data_example/physicochemical_data/SMILES_data ./data_example/physicochemical_data/FGT_data/
```

For additional SMILES-to-FGT conversions, you may also refer to `decode_test.py` or `encode_test.py` in the `tools/SMILES_to_FGT` directory.



---

### Chemprop Training Pipeline

The `chemprop_train` directory provides a complete workflow for downloading ChEMBL assay data and training Chemprop predictors:

* **`download_chembl.py`** automatically retrieves and preprocesses assay-specific bioactivity data from ChEMBL.
* **`train_chemprop.py`** launches an automated, multi-round Chemprop training pipeline that builds a dedicated predictor for each target.

This pipeline progressively increases model complexity to obtain a robust structure–property predictor. Early rounds use a lightweight architecture (hidden size 600, 50 epochs), while later rounds automatically scale the hidden size (up to 1800), extend training length (up to 200 epochs), enable ensembles of up to 5 models, and incorporate 2D RDKit features.

ChEMBL activity values are first transformed using log10 scaling, followed by StandardScaler normalization (the scaler is saved for inverse transformation during inference). For each training round, assay data is split into a 90% training/validation set and a 10% held-out test set, with the 90% portion internally separated by Chemprop using caffold-balanced splitting. Models are trained using the default Adam optimizer with adaptive early stopping (patience increased from 5 to 20). The pipeline automatically monitors performance on the held-out test set and iterates training until the predictor reaches a Pearson correlation above 0.75.

---


### Property Generation and Success Rate Testing

The `generate` directory provides scripts for evaluating the success rate of molecular optimization across both physicochemical and biological properties.

* To test the success rate for **physicochemical properties** (e.g., LogP optimization), run:

  ```bash
  bash run_physicochemical.sh
  ```

* To test the success rate for **biological properties** using trained Chemprop predictors, run:

  ```bash
  bash run_biological.sh
  ```

These scripts automatically perform property-guided generation using DrugLLM, compute success rates based on predefined criteria, and save the evaluation results for further analysis.

---

### Note on `SMILES_to_FGT-v2`

The `tools` directory also contains an updated implementation, **`SMILES_to_FGT-v2`**, which provides **higher accuracy** for molecular encoding (SMILES → FGT) and decoding (FGT → SMILES).


