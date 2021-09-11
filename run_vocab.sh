WIKISPLIT_DIR=/data/pengwei/fourth/laser_data/wiki-split
# Preprocessed data and models will be stored here.
OUTPUT_DIR=/data/pengwei/fourth/laser_data/wiki-split/output
# Download the pretrained BERT model:
# https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
BERT_BASE_DIR=/data/pretrained_models/cased_L-12_H-768_A-12

### Optional parameters ###

# If you train multiple models on the same data, change this label.
EXPERIMENT=wikisplit_experiment
python phrase_vocabulary_optimization.py \
  --input_file=${WIKISPLIT_DIR}/train.tsv \
  --input_format=wikisplit \
  --vocabulary_size=500 \
  --max_input_examples=1000000 \
  --output_file=${OUTPUT_DIR}/label_map.txt
