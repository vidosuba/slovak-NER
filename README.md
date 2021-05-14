# Contextualized Language Model-based Named Entity Recognition in Slovak Texts

Named Entity Recognition (NER) is one of the fundamental tasks in Natural
Language Processing (NLP), with English state-of-the-art approaches generally
utilizing neural models. The currently available NER classifiers for Slovak
texts are either rule- and vocabulary-based systems or employ multilingual
Contextualized Language Models. Both of these show poor performance
compared to the language-specific Deep Contextualized Language Models,
even in low-resource languages, such as Slovak.

## Goals
- Review the existing NER models and the datasets they are trained on
- Refine and extend the annotated (Slovak) datasets for NER
- Take advantage of transfer learning between multilingual and Slovak NER
models
  
## Links
 - [Presentation](./diploma/project_seminar_1.pdf) for Project Seminar 1
 - [Structure of thesis](./diploma/dplm.pdf)
 - [Prodigy](http://http://davidsuba.tk:8888/)

## Used Publications
 - [MasakhaNER: Named Entity Recognition for African Languages](https://arxiv.org/pdf/2103.11811.pdf)
 - [Benchmarking Pre-trained Language Models for Multilingual NER: TraSpaS at the BSNLP2021 Shared Task.  InProceedings of the 8th Workshop on Balto-Slavic Natural Language Process-ing](https://www.aclweb.org/anthology/2021.bsnlp-1.13.pdf)
 - [DAGA: Data Augmentation with a Generation Approach for
Low-resource Tagging Tasks](https://www.aclweb.org/anthology/2020.emnlp-main.488.pdf)

## Progress
 - find all existing Slovak NER datasets
 - clean WikiANN dataset
 - train Trankit model on WikiANN
 - train Spacy model on WikiANN
 - deploy and test prodigy tool