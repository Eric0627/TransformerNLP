# Transformer for Natural Language Processing

## Overview

This project involves implementing (from scratch) and experimenting with different parts of the transformer architecture. We work with speeches from three American politicians to build classification and language modeling systems.

## Data Description

The dataset consists of speeches from three American politicians:
- **0:** Barack Obama
- **1:** George W. Bush  
- **2:** George H. Bush

### Files Structure
- **Classification Task:**
  - Train: `train_CLS.tsv` (tab-separated: label + tab + speech segment)
  - Test: `test_CLS.txt`
- **Language Modeling Task:**
  - Train: `train_LM.txt`
  - Test: `test_LM_obama.txt`, `test_LM_wbush.txt`, `test_LM_hbush.txt`

## Prerequisites

### Required Libraries
- PyTorch
- NLTK (for tokenization)
- Standard Python libraries

### Starter Code Files
- `dataset.py` - PyTorch Dataset classes for classification and language modeling
- `tokenizer.py` - Simple word-level tokenizer using NLTK
- `utilities.py` - Helper functions for attention matrix sanity checks and visualization
- `main.py` - Default parameters and example usage
- `transformer.py` - **Empty file where you'll implement your transformer components**

### Positional Encoding
For the base implementation, use simple absolute positional embeddings:
- Two embedding tables: one for tokens, one for positions
- Add positional embeddings to token embeddings before feeding into transformer blocks

---

## Part 1: Encoder Trained With Classifier

We implemented a transformer encoder with a feedforward classifier for politician speech classification. We trained both components jointly from scratch without pretraining.

### Part 1.1: Encoder Implementation
- Implemented transformer encoder following hyperparameters in `main.py`
- Output: sequence of embeddings for each input word
- Used **mean pooling** across the sequence dimension to provide embeddings to the classifier

### Part 1.2: Feedforward Classifier Implementation  
- Simple feedforward network with one hidden layer
- Input: mean-pooled encoder embeddings
- Output: predictions for which politician spoke the speech segment

### Part 1.3: Joint Training
- Passed input through the encoder, generated embeddings, and fed them to the classifier
- Computed loss between predictions and true labels
- Updated both encoder and classifier weights via backpropagation
- Trained both components simultaneously

### Part 1.4: Sanity Checks
- Used `utilities.py` helper function to verify attention implementation
- Checked that the attention matrix rows sum to 1

### Part 1.5: Evaluation
Tested on `test_CLS.txt`.

**Performance:**
| Metric | Value |
|--------|--------|
| Number of Parameters | 2,158,155 |
| Vocabulary Size | 30,522 |
| Epoch 1 Train Loss | 1.0763 |
| Epoch 15 Train Loss | 0.1263 |
| Epoch 15 Test Accuracy | 86.13% |

---

## Part 2: Pretraining Decoder Language Model

We implemented a GPT-like transformer decoder for autoregressive language modeling and evaluated perplexity on speeches given by different politicians.

### Part 2.1: Decoder Implementation
- Similar to encoder but used **masked self-attention** 
- Prevented the model from seeing future tokens during training
- Feedforward specifications:
  - Hidden dimensionality: 100
  - Activation function: ReLU
  - Input/output dimensionality: `n_embed = 64`

### Part 2.2: Decoder Pretraining
- Task: Predict the next word given previous words in sequence
- Output: Probability distribution over vocabulary
- Loss: Cross-entropy between predictions and true next words
- Training limit: **500 iterations (batches)** with batch size 16, block size 32
- Total tokens processed: ~256,000

### Part 2.3: Sanity Checks
- Used `utilities.py` to verify attention implementation

### Part 2.4: Evaluation
Tested on all three politician test sets and reported perplexity.

**Performance:**
| Metric | Value |
|--------|--------|
| Number of Parameters | 863,243 |
| Vocabulary Size | 5,755 |
| Step 500 Train Perplexity | 169.3392 |
| Step 500 Obama Perplexity | 367.9337 |
| Step 500 H. Bush Perplexity | 419.1233 |
| Step 500 W. Bush Perplexity | 482.0752 |

## Implementation Notes

### Key Architecture Components to Implement
- Multi-head attention mechanisms
- Layer normalization
- Feedforward networks
- Positional embeddings
- Masked attention (for decoder)

### Training Tips
- Monitor attention matrix properties during development
- Verify masking implementation by checking the train perplexity behavior
- Use the provided hyperparameters initially for comparable results

## Running
This is an instruction on how to run the project. 

### Prerequisites
```
torch
numpy
matplotlib
nltk
transformers # BertTokenizer
```
### Commands
```
# Part 1: Encoder Trained With Classifier
python main.py --part 1
# Part 2: Pretraining Decoder Language Model
python main.py --part 2
# Part 3 [Exploration]: Decoder with Masked Multi-query Attention
python main.py --part 3
# Get visualization (the figures in the report) by appending the following option to the commands above 
--visualization true
```
