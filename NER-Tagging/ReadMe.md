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


Training the Models, generating predictions and evaluating Model Performance
    To train the BiLSTM NER model with both embeddings and generate predictions on the dev/test sets using the trained models, 
    run straight thru the notebook and run eval in a cell for both task 1 and task 2 dev files
        

Evaluating Model Performance
    To evaluate the model's performance on the dev set, use the provided evaluation script with Perl installed:
        bash
        python eval.py -p ..\data\lstm-data\preds\dev_preds_glove_noscheduler -g ../data/lstm-data/dev


Description of Solution
    The models are built using PyTorch, with the architecture specified in the tasks. The custom embedding model, 
    embeddings are learned from scratch. For the GloVe embedding model, the embedding layer is initialized with
    pre-trained GloVe embeddings, adapted for case sensitivity.


Key hyperparameters:
    Optimizer: SGD with momentum 0.9, learning rate 0.05 (custom) and 0.1 (GloVe), weight decay 0.0001
    Batch size: 32 and 16
    Number of epochs: Up to 150 with early stopping based on validation F1 score
    Patience for early stopping: 10 (custom) and 150 (GloVe)
    For Loss function passed in the inverse frequency weights for the classes to try to manage class imbalances and
    ignore index to avoid padding indices being used in loss function

    These hyperparameters were chosen based on empirical performance on the development set using a custom gridsearch function. 
    The models are trained on the provided training data and evaluated on the development data to measure precision, recall, and F1 score.


Questions: 
	Task 1: 
	processed 51578 tokens with 5942 phrases; found: 6164 phrases; correct: 5282.
    accuracy:  97.56%; precision:  85.69%; recall:  88.89%; FB1:  87.26
                LOC: precision:  90.73%; recall:  94.28%; FB1:  92.47  1909
                MISC: precision:  78.23%; recall:  81.45%; FB1:  79.81  960
                ORG: precision:  80.65%; recall:  82.70%; FB1:  81.66  1375
                PER: precision:  88.02%; recall:  91.75%; FB1:  89.85  1920

    Task 2: 
    processed 51578 tokens with 5942 phrases; found: 6133 phrases; correct: 5288.
    accuracy:  97.56%; precision:  86.22%; recall:  88.99%; FB1:  87.59
                LOC: precision:  93.39%; recall:  93.03%; FB1:  93.21  1830
                MISC: precision:  78.51%; recall:  81.24%; FB1:  79.85  954
                ORG: precision:  79.97%; recall:  84.27%; FB1:  82.06  1413
              PER: precision:  87.81%; recall:  92.29%; FB1:  89.99  1936

Command to run python code: 
    C:/Users/amant/anaconda3/envs/pytorch_dl/python.exe ./main.py