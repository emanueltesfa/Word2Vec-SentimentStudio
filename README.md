Dataset Overview: The dataset comprised Amazon reviews across various product categories, which we balanced at 250K 
entries with 50K instances per rating score. We transformed the ratings into ternary sentiment labels: 
ratings above 3 as positive (class 1), below 3 as negative (class 2), and ratings of 3 as neutral (class 3). We 
split the dataset into an 80/20 ratio for training and testing, respectively. Furthermore, we split the ternary 
data into binary data with 100k negative reviews and 100k positive reviews, named filtered_reviews.  Data Preprocessing: Given the textual nature of the data, preprocessing was paramount and very similar to HW1. We 
implemented a pipeline including lowercasing, HTML tag removal, contraction expansion, stop-word 
removal, and lemmatization. This standardization ensured the removal of noise and normalization of the 
text, providing a cleaner dataset for model input. Feature Extraction: Two sets of Word2Vec features were extracted: one using the pretrained "word2vec-google-news-300" 
model and another trained on our custom dataset with a 300-dimensional space, a window size of 11 and 
minimum count of 10 words. We observed that the semantic similarity between word pairs such as 'ocean' 
and 'sea' was well captured by both models, with nuanced differences highlighting the potential impacts 
of domain-specific training. In our case, the pretrained models similarity score between Sea and Ocean 
was lower than the custom model I trained, thus showing that similarities between vectors were stronger 
in my model. But the pretrained model does a better join of building relationships between words. Model Selection and Training: We utilized a Perceptron and an SVM for initial experiments, using the average Word2Vec vectors as 
features. Due to SVM extremely long training times, the max_iter parameter was dropped down to 200 to 
ensure we could run it multiple times. This led to poor performance of SVM. The Custom model trained 
on the review data again performed better (for both Perceptron and SVM) than Google's pretrained 
model. These models served as a baseline, which we compared against a feedforward neural network 
(FNN) with two hidden layers and a simple two-layer CNN, with Gelu activation functions since we are 
working with language inputs and utilized dropout to reduce overfitting. both designed to accommodate 
the dimensions of our Word2Vec features, the pretrained features and also implementing feature 
concatenation. Furthermore, The FNN and CNN models were trained with both binary and ternary 
outputs. Observations and Model Adjustments: Our observations indicated that while traditional models provided a reasonable baseline, the FNN and 
CNN architectures delivered significant improvements in accuracy. This was attributed to their capacity to 
capture non-linear relationships and hierarchical features in the data. Hyperparameter tuning, such as 
adjusting learning rates and the number of epochs, further refined model performance. 7. Results: The deep learning models consistently outperformed the baseline models. Among them, in ternary 
classification, CNN demonstrated a higher accuracy, potentially due to its ability to capture local 
dependencies and patterns within the review texts. Notably, the custom Word2Vec features marginally 
outperformed the pretrained-trained features, indicating the issue of overgeneralization and robustness of 
widely applicable language representations in task specific domains such as reviews. 
#### Conda requirements after running 'conda list --export' #### beautifulsoup4=4.12.2=py312haa95532_0 brotli-python=1.0.9=py312hd77b12b_7 ca-certificates=2023.12.12=haa95532_0 certifi=2023.11.17=py312haa95532_0 cffi=1.16.0=py312h2bbff1b_0 gensim=4.3.2=py312hc7c4135_0 importlib-metadata=7.0.1=pyha770c72_0 importlib_metadata=7.0.1=hd8ed1ab_0 intel-openmp=2023.1.0=h59b6b97_46320 mkl=2023.1.0=h6b88ed4_46358 mkl-service=2.4.0=py312h2bbff1b_1 mkl_fft=1.3.8=py312h2bbff1b_0 mkl_random=1.2.4=py312h59b6b97_0 numpy=1.26.3=py312hfd52020_0 numpy-base=1.26.3=py312h4dde369_0 openssl=3.2.1=hcfcfb64_0 pandas=2.1.4=py312hc7c4135_0 pip=23.3.1=py312haa95532_0 pycparser=2.21=pyhd3eb1b0_0 python=3.12.1=h1d929f7_0 pytorch=2.2.0=py3.12_cuda12.1_cudnn8_0 pytorch-cuda=12.1=hde6ce7c_5 scikit-learn=1.3.0=py312hc7c4135_2 scipy=1.11.3=py312h3d2928d_0 setuptools=68.2.2=py312haa95532_0 six=1.16.0=pyh6c4a22f_0 smart_open=5.2.1=py312haa95532_0 sqlite=3.41.2=h2bbff1b_0 tk=8.6.12=h2bbff1b_0 tqdm=4.65.0=py312hfc267ef_0 
wheel=0.41.2=py312haa95532_0 xz=5.4.5=h8cc25b3_0 yaml=0.2.5=he774522_0 