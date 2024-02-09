# Amazon Review Sentiment Analysis

## Dataset Overview

Our dataset consists of Amazon reviews across various product categories. It is balanced to contain 250,000 entries, with 50,000 instances per rating score. Ratings were transformed into ternary sentiment labels: above 3 as positive (class 1), below 3 as negative (class 2), and ratings of 3 as neutral (class 3). The dataset was split into an 80/20 ratio for training and testing, respectively. Additionally, we created `filtered_reviews` with 100,000 negative and 100,000 positive reviews for binary sentiment analysis.

## Data Preprocessing

The preprocessing pipeline was akin to HW1, involving lowercasing, HTML tag removal, contraction expansion, stop-word removal, and lemmatization. This standardization process ensured the text was normalized, reducing noise and preparing the dataset for model input.

## Feature Extraction

We extracted Word2Vec features using two approaches: one with the "word2vec-google-news-300" pretrained model and another with our custom-trained model. Our custom model showcased a stronger semantic similarity between word pairs such as 'ocean' and 'sea', though the pretrained model was more adept at constructing word relationships.

## Model Selection and Training

We experimented with a Perceptron and an SVM, using the average Word2Vec vectors as features. Given the SVM's lengthy training times, we limited `max_iter` to 200. We also implemented a feedforward neural network (FNN) with two hidden layers and a simple two-layer CNN, both utilizing Gelu activation functions and dropout to mitigate overfitting. These models were evaluated with both binary and ternary outputs.

## Observations and Model Adjustments

While traditional models established a reasonable baseline, our FNN and CNN architectures substantially enhanced accuracy, capturing non-linear relationships and hierarchical data features. Hyperparameter tuning further improved model performance.

## Results

Deep learning models consistently surpassed the baseline models. The CNN, in particular, excelled in ternary classification due to its capability to capture local dependencies within the texts. Our custom Word2Vec features slightly outperformed the pretrained ones, underscoring the value of domain-specific training.

## Conda Requirements

After running `conda list --export`, the following requirements were identified for our environment:

```plaintext
beautifulsoup4=4.12.2=py312haa95532_0
brotli-python=1.0.9=py312hd77b12b_7
ca-certificates=2023.12.12=haa95532_0
certifi=2023.11.17=py312haa95532_0
cffi=1.16.0=py312h2bbff1b_0
gensim=4.3.2=py312hc7c4135_0
importlib-metadata=7.0.1=pyha770c72_0
importlib_metadata=7.0.1=hd8ed1ab_0
intel-openmp=2023.1.0=h59b6b97_46320
mkl=2023.1.0=h6b88ed4_46358
mkl-service=2.4.0=py312h2bbff1b_1
mkl_fft=1.3.8=py312h2bbff1b_0
mkl_random=1.2.4=py312h59b6b97_0
numpy=1.26.3=py312hfd52020_0
numpy-base=1.26.3=py312h4dde369_0
openssl=3.2.1=hcfcfb64_0
pandas=2.1.4=py312hc7c4135_0
pip=23.3.1=py312haa95532_0
pycparser=2.21=pyhd3eb1b0_0
python=3.12.1=h1d929f7_0
pytorch=2.2.0=py3.12_cuda12.1_cudnn8_0
pytorch-cuda=12.1=hde6ce7c_5
scikit-learn=1.3.0=py312hc7c4135_2
scipy=1.11.3=py312h3d2928d_0
setuptools=68.2.2=py312haa95532_0
six=1.16.0=pyh6c4a22f_0
smart_open=5.2.1=py312haa95532_0
sqlite=3.41.2=h2bbff1b_0
tk=8.
