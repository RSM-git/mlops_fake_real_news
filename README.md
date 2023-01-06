## Machine Learning Operations project description - NLP with fake and real news articles

Casper Brun Pedersen - 
Marcus Presutti -
Rasmus Steen Mikkelsen - 
Victor Tolsager Olesen -

### Overall goal of the project

The goal of the project is to develop a MLOps pipeline for classifying news articles as either fake or real using Transformers.

### What framework are you going to use (PyTorch Image Models, Transformers, PyTorch-Geometrics)

The model will be trained on text data, which prompts the use of the Transformers framework.

### How do you intend to include the framework in your project?

The transformers framework contains a multitude of pretrained models. We intend to fine-tune the pretrained ALBERT transformer

### What data are you going to run on? (initially, may change)

We will be using a Kaggle dataset, specifically the [Fake and real news dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset). The dataset consists of two `.csv` files `Fake.csv` and `True.csv`, which will be merged into a single dataset, each sample initially has four attributes: `title`, `text`, `subject`, and `date`, and will naturally get a label attribute based on which file the sampled was initially in.

### What deep learning models do you expect to use?
