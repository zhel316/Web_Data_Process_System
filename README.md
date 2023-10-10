# Web_Data_Process_System

## Introduction
This project aims to construct a pipeline capable of recognizing named entities from web pages, identifying links that point to knowledge bases (KBs), and extracting the relationships between them. The overall task can be broken down into the following steps:
1. Parsing HTML to plain text
2. Named Entity Recognition
3. Entity Linking
4. Relation Extraction

## Parsing HTML to plain text
Input WARC files are stored in the `/data` directory. The `html2text.py` script is responsible for this conversion.

## Named Entity Recognition (NER)
Named entities are distinct terms presented in the text, such as a person’s name, a geographic location, or an organization's name. NER is a crucial task in natural language processing that involves identifying and classifying these named entities within a given text. We leverage Spacy, an open-source natural language processing library, for NER implementation. The specific code for this can be found in `NER.py`.

## Entity Linking
Entity linking, also known as entity disambiguation, is the process of identifying and associating entities mentioned in a text with a corresponding entity in a knowledge base or website that provides more detailed information about the entity. We've adopted the WSD method for entity disambiguation, which can be found in `WSD.py`.

## Relation Extraction
A Convolutional Neural Network (CNN) model is employed to extract relationships between entities. Apart from the CNN model, we also tried R-Bert model. R-BERT model [3] is a derivative of Bert’s model for relationship extraction, with a simple but powerful structure.
