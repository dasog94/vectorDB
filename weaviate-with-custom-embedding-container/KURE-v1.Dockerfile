# refer https://github.com/weaviate/t2v-transformers-models?tab=readme-ov-file#custom-build-with-any-huggingface-model
FROM semitechnologies/transformers-inference:custom
RUN MODEL_NAME=nlpai-lab/KURE-v1 ./download.py