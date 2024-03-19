README for Running NER Model Training and Evaluation

Requirements
    Python 3.8 or newer
    PyTorch 1.8.1 or newer
    NumPy
    tqdm
    scikit-learn

bash
    pip install torch==1.8.1 numpy tqdm scikit-learn torch 

Files Description
    notebooks/NamedEntityDescription.ipynb: The main script that contains the model definition, training, and evaluation logic.
    data/: Directory containing the train, dev, and test data files.
    glove.6B.100d.txt: GloVe embeddings file.
    notebooks/ckpts/: Directory where trained model checkpoints will be saved.

Training the Models and generating preidctions
    To train the BiLSTM NER model with both embeddings and generate predictions on the dev/test sets using the trained models, run straight thru the notebook and it will print the location of the model 
        

Evaluating Model Performance
    To evaluate the model's performance on the dev set, use the provided evaluation script with Perl installed:
        bash
        python eval.py -p ..\data\lstm-data\preds\dev_preds_glove_noscheduler -g ../data/lstm-data/dev