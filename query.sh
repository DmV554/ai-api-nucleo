#!/bin/bash

curl -X POST "http://127.0.0.1:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Is any information about the origin of the covid?",
    "collection_name": "covid_qa",
    "strategy": "naive",
    "top_k": 3
  }'
