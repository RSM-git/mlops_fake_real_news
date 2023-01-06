## Machine Learning Operations project description - NLP with fake and real news articles

Casper Brun Pedersen - s204119 <br>
Marcus Presutti - s204122 <br>
Rasmus Steen Mikkelsen - s204135 <br>
Victor Tolsager Olesen - s204141

### Overall goal of the project

The goal of the project is to develop an MLOps pipeline for classifying news articles as either fake or real using Transformers.

### What framework are you going to use (PyTorch Image Models, Transformers, PyTorch-Geometrics)

The model will be trained on text data, which prompts the use of the Transformers framework.

### How do you intend to include the framework in your project?

The transformers framework contains a multitude of pretrained models. We intend on fine-tuning the pretrained ALBERT transformer

### What data are you going to run on? (initially, may change)

We will be using a Kaggle dataset, specifically the [Fake and real news dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset). The dataset consists of two `.csv` files `Fake.csv` and `True.csv`, which will be merged into a single dataset. Each sample initially has four attributes: `title`, `text`, `subject`, and `date`, and will naturally get a label attribute based on which file the sample was initially in.

### What deep learning models do you expect to use?

We have intentions of using the pretrained [ALBERT](https://huggingface.co/docs/transformers/model_doc/albert) from huggingface, which will be fine-tuned using the fake and real news dataset. ALBERT performs parameter reduction techniques to lower memory consumption and increase training speed of the original [BERT](https://huggingface.co/docs/transformers/model_doc/bert) model.
