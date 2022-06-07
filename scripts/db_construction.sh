#!/bin/bash

pipenv run python -m spacy download en_core_web_trf
pipenv run python ./src/db_construction.py
