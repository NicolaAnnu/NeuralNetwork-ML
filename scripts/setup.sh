#!/usr/bin/env bash


python -m venv .env
source .env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
