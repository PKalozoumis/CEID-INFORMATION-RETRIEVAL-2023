# 🌐 CEID-INFORMATION-RETRIEVAL-2023

This project was made for the **7th semester** undergraduate elective course **"Information Retrieval"** of the **Computer Engineering & Informatics Department (CEID), University of Patras**.

The purpose of this project is to implement a **search engine** on a collection of 1239 documents from the **Cystic Fibrosis** collection. An **inverted index** is constructed from the collection and stored as a file, which is then used by a custom implementation of the **Vector Space Model** to retrieve the most relevant documents for each of the 20 test queries. In addition to this classic model, a pre-trained neural information retrieval model, **ColBERT**, is also used to calculate word embeddings for each token in the documents and queries, thus capturing their semantic similarity more effectively. The two models are finally compared using classic metrics, such as **Precision, Recall, NDCG**, etc.

# Dataset
The dataset consists of 1239 documents from the [Cystic Fibrosis](https://people.ischool.berkeley.edu/~hearst/irbook/cfc.html) collection. Before doing anything else, make sure to extract this dataset from the `dataset.tar.gz` file by running the following command:
```shell
tar -xvzf dataset.tar.gz
```
You should be seeing 3 folders: `original_dataset`, `json_docs` and `json_queries`. ColBERT expects the dataset to be in a specific format, as it appears in the JSON folders. This is handled by the `dataset.py` module, if those directories don't already exist.

# Setup
Fist you need to clone this repository to your local machine. Note that for this project, [Stanford's implementation of ColBERT](https://github.com/stanford-futuredata/ColBERT) was used, which is also its own GitHub repository. It integrates into this project as a Git submodule. To clone both repositories simultaneously, run the following command:
```shell
git clone --recurse-submodules git@github.com:PKalozoumis/CEID-INFORMATION-RETRIEVAL-2023.git
```

### Setting up the conda environment
For ColBERT to work it needs a specific set of packages, which are described in a conda environment that it provides. First, install conda by following the [official installation guide](https://www.anaconda.com/docs/getting-started/installation#installation). The Miniconda distribution is sufficient for this project. The commands look something like this. Make sure say "yes" to initialization when asked:

```shell
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
```shell
bash ~/Miniconda3-latest-Linux-x86_64.sh
```

Depending on whether you are on a GPU or CPU-only environment, you want to choose either the `conda_env.yml` or `conda_env_cpu.yml` file respectively. In the case of a GPU, run:
```shell
conda env create -f ColBERT/conda_env.yml
```
Keep in mind this can be a resource-intensive process, taking up a lot of time and RAM. If you're facing problems, I suggest changing conda's default dependency solver to Mamba before creating the environment, which can be faster and more resource-friendly:
```shell
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

Once it finishes creating the environment, activate it. If everything goes well, you should be seeing `(colbert)` in front of your username:
```shell
conda activate colbert
```

In addition to ColBERT's dependencies, make sure to also install the following packages, which are required by the rest of the project:
```shell
pip install matplotlib
pip install openpyxl
```

### Downloading the pre-trained checkpoint
We use ColBERT to index and search our dataset, but we do not want to retrain it. Instead, we download the provided checkpoint, which was trained on the MS MARCO Passage Ranking task:
```
wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz
```
Unzip it:
```
tar -xvzf colbertv2.0.tar.gz
```

# How to Run
While you're in the ColBERT virtual environment, run the `main.py` script. If this is the first time running it, make sure to include the `--index-first` option to make ColBERT index our documents:
```shell
python main.py --index-first
```