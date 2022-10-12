<a name="readme-top"></a>
<!-- ABOUT THE PROJECT -->
# Fine-Tuning a Sentence Transformer on Ukrainian Data

Ukrainian semantic similarity using sentence_transformers

### Ukrainian: A Low-Resource Language

A crucial part of Natural Language Processing (NLP) is having access to expert-annotated datasets in order to train models. Languages lacking in such corpora are known as low-resource languages.

It is necessary to support low-resource languages’ use in NLP so that these languages can be preserved and expanded digitally. Within this project, I fine-tuned a sentence transformer model on a Ukrainian dataset so as to study potential solutions for NLP with a lack of annotated data.

<!-- FILES -->
## Files

### uk_en_semantic_similarity.py

I was interested in determining the semantic similarity between Ukrainian and English sentences. The pretrained sentence transformer fine-tuned was distiluse-base-multilingual-cased-v2. It is a “[multilingual knowledge distilled version of multilingual Universal Sentence Encoder](https://www.sbert.net/docs/pretrained_models.html)”, supporting 50+ languages.

#### Dealing with the absence of labelled ukr datasets

I was unable to obtain a Ukrainian dataset with expert-annotated similarity scores. Instead, I used the Tatoeba Translation Challenge dataset (Helsinki-NLP/tatoeba-mt on Hugging Face). I specifically chose tatoeba because it has pairs of sentences from the same language, which contain minor differences in phrasing.

Since the sentences in the pairs are nearly identical I set the scores to 1 for each pair in the dataset (though the actual scores of these pairs would not be 1 if annotated expertly). I then duplicated the dataset and shuffled the pairings, setting the shuffled sentence pairs’ scores to 0, thus labelling those pairs as “dissimilar”. 

#### Evaluation

The evaluator used outputs the Spearman’s rank correlation between cosine similarity scores of the provided dataset and the scores calculated from the model (correlation value between 0 and 1, where: 0 = no correlation, and 1 = high correlation).

### calculate_similarities.ipynb

This notebook generates a matrix of cosine similarities between each sentence in a given list of sentences.

### ./notebook

Contains a notebook version of uk_en_semantic_similarity.py.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Kate Richards - [in/katemrichards](https://www.linkedin.com/in/katemrichards/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
