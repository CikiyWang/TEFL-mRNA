# TEFL-mRNA

**TEFL-mRNA** is a hybrid deep learning model that integrates Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) to predict **Translation Efficiency (TE)** from **full-length mRNA sequences**.

## 🔬 Overview

Translation Efficiency (TE) plays a key role in post-transcriptional regulation and gene expression. Understanding how TE is encoded in mRNA sequences can illuminate mechanisms of translational control and enable better design of synthetic genes and RNA therapeutics.

**TEFL-mRNA** was developed to address this challenge by leveraging a hybrid deep learning architecture:

- 🧠 **CNN layers** capture local sequence motifs and short-range dependencies.
- 🔁 **RNN (e.g., LSTM or GRU) layers** model long-range dependencies and global sequence context.
- 📊 The model takes **full-length mRNA sequences** as input and predicts a continuous TE value.

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/TEFL-mRNA.git
cd TEFL-mRNA
pip install -e .

conda create -n tefl_env python=3.8
conda activate tefl_env
pip install -r requirements.txt

**## 🧪 Usage**
**Train the model**
```bash
python experiments/train_model.py

**Predict TE for new mRNAs sequence**
```bash
python predict_te.py --input sequences.fasta
