# DrugLLM

Large Language Models (LLMs) have transformed machine learning, particularly in few-shot learning and reasoning, showing impressive results across natural language processing and computer vision. Yet, in specialized fields like biology and chemistry, traditional LLMs face significant challenges, especially in understanding complex relationships between molecular structures and pharmacochemical properties. These limitations hinder effective few-shot learning for small-molecule generation and optimization within drug discovery.

We introduce **DrugLLM**, a groundbreaking LLM tailored specifically for molecular optimization. DrugLLM leverages **Group-based Molecular Representation (GMR)**, a method that tokenizes molecular structures to better align with LLM training and addresses inconsistencies inherent in SMILES encoding. With a unique pre-training strategy, DrugLLM iteratively refines molecular structures through few-shot adjustments, guiding each modification towards a specific pharmacological goal. In comprehensive computational experiments, DrugLLM has achieved state-of-the-art results in molecular generation. When applied to the optimization of HCN2 inhibitors, DrugLLM successfully generated bioactive compounds verified in laboratory settings, showcasing its potential to drive AI-enabled molecule optimization and accelerate drug discovery.

For more information, please refer to our paper on [arXiv:2405.06690](https://arxiv.org/abs/2405.06690).

## Usage Guide

### Model Download
The DrugLLM model is hosted on Hugging Face and can be downloaded from [here](https://huggingface.co/ziyanglichuan/DrugLLM). After downloading, place the model files in the `model` directory.

### Training Dataset
The training dataset and supporting experimental data are available at [https://osf.io/f6yqn/](https://osf.io/f6yqn/).

### Testing Dataset
A sample test dataset is included in the `GMR_test` directory.

### Installation
To install the required dependencies, execute the following command:

```bash
pip install -r requirements.txt
```

> **Note**: This project is best supported on Python 3.9.

### Data Preprocessing
To convert molecular data from SMILES format to Group-based Molecular Representation (GMR), run the command below:

```bash
python jsonSmiles_to_GMR.py <input_directory> <output_directory>
```

- **Parameters**:
  - `<input_directory>`: Path to the directory with SMILES data files.
  - `<output_directory>`: Path where the converted GMR data will be saved.

**Example**:
```bash
python jsonSmiles_to_GMR.py ./SMILES_data ./GMR_data/
```

For additional SMILES-to-GMR conversions, you may refer to `decode_test.py` or `encode_test.py` located in the `tools/SMILES_to_GMR` directory.

### Property Generation and Success Rate Testing
To generate molecular structures optimized for the LogP property and evaluate the generation success rate, use the command:

```bash
python GMR_generate.py <input_file>
```

- **Parameter**:
  - `<input_file>`: Path to the file containing GMR data for testing molecular generation and success rate evaluation.

**Example**:
```bash
python GMR_generate.py ../data_example/GMR_data/GMR_LogP_test.json
```
