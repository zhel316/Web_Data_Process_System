# Web_Data_Process_System

## Introduction
This project aims to implement a pipeline that recognizes named entities from web pages, finds links referred to KBs, and extracts the relation between them. The task can be split into the following parts:
1. Parsing HTML to plain text
2. Named Entity Recognition
3. Entity Linking
4. Relation Extraction

## Parsing HTML to plain text
The input WARC files are stored in `/data`. `html2text.py` implements this function.

## Named Entity Recognition (NER)
Named entities are entities in a text with a unique name, such as a person’s name, a place name, or an organization name. NER is an important task in natural language processing that involves identifying and labelling named entities in a text. We achieved NER by using Spacy, an open-source natural language processing library.
## Entity Linking

## Relation Extraction
