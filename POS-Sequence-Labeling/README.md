# POS Tagging Notebook Overview

This notebook presents a comprehensive guide to building a Part-of-Speech (POS) tagging system using Hidden Markov Models (HMMs). The process encompasses data preprocessing, vocabulary creation, model training, and decoding strategies including Greedy and Viterbi algorithms.

## Features

- **Vocabulary Creation:** Establish a vocabulary from training data, replacing infrequent words with a special token `<unk>` to handle unknown words effectively.
- **HMM Model Training:** Calculate emission and transition probabilities from the training corpus to model the language and tagging patterns.
- **Decoding Strategies:** Implement Greedy and Viterbi decoding algorithms to predict the sequence of tags for new sentences.

## Workflow

1. **Data Preprocessing:** Load and preprocess the training, development, and test datasets, extracting word and tag sequences.
2. **Vocabulary Management:** Implement a threshold-based (N_threshold = 3) approach to manage infrequent words, enhancing the model's ability to generalize.
   - Total vocabulary size: 12405 
   - Total word occurrences: 76948215 
3. **Model Training:**
   - Calculate state (tag) probabilities, transition probabilities between tags, and emission probabilities of words given tags.
   - Save the trained HMM parameters for future use.
   - Parameters: 
      - 2025 transition probabilities  
      - 761400 emission probabilities 
4. **Decoding:**
   - **Greedy Decoding:** For each word in a sentence, choose the tag with the highest probability, given the previous tag and the word itself.
   - **Viterbi Decoding:** Employ a dynamic programming approach to find the most probable tag sequence for the entire sentence.

## Results 
   - **Greedy Accuracy**: 92.68% 
   - **Viterbi Accuracy:** 94.36% 

## Files and Directories

- `vocab-data/`: Contains training, development, and test datasets.
- `outputs/`: Directory where the vocabulary file, HMM parameters, and prediction outputs are saved.
- `eval.py`: Evaluation script to compute the accuracy of the tag predictions.

## Requirements

- Python 3.x
- Pandas
- NumPy
- tqdm
- json
Ensure all dependencies are installed to run the notebook without issues.

## Running the Notebook

1. Ensure that the `vocab-data/` directory is populated with the `train`, `dev`, and `test` files.
2. Execute the notebook cells in order, from top to bottom, to perform vocabulary creation, model training, and tag prediction.
3. Use the `eval.py` script to evaluate the predictions against the gold standard tags provided in the `dev` and `test` datasets.

## Evaluation

Run the evaluation script with the following command:

```bash
python eval.py -p <path_to_predictions_file> -g <path_to_gold_standard_file>



 

 
 
This solution presents a sophisticated approach to Part-of-Speech (POS) tagging utilizing Hidden Markov 
Models (HMMs), a probabilistic framework well-suited for sequence labeling tasks in natural language 
processing. The process begins with data preprocessing, where a vocabulary is constructed from the 
training corpus, with infrequent words being replaced by a special token to manage unknown words 
effectively. This step is crucial for improving the model's ability to generalize to unseen data. The core of 
the solution lies in calculating the emission probabilities of words given their tags and the transition 
probabilities between tags, which are derived from the training data. These probabilities form the 
backbone of the HMM. 
For decoding—the process of predicting tags for new sentences—two strategies are implemented: 
Greedy decoding and the Viterbi algorithm. The Greedy approach selects the most probable tag for each 
word sequentially, considering the previous tag and the word itself, which, while fast, may not always 
yield the best results. The Viterbi algorithm, on the other hand, employs dynamic programming to find 
the most probable sequence of tags for the entire sentence, ensuring a globally optimal solution. This 
comprehensive methodology not only highlights the practical application of HMMs in POS tagging but 
also demonstrates the balance between computational efficiency and accuracy in predictive modeling 
tasks. The solution, encapsulated in a well-structured Jupyter notebook, provides a step-by-step guide 
from vocabulary creation and model training to tag prediction, showcasing a robust framework for 
tackling POS tagging challenges. 
 
