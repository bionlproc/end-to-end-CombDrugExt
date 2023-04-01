# End-to-End n-ary Relation Extraction for Combination Drug Therapies
This is a code repository for our paper at IEEE ICHI 2023 titled: End-to-End n-ary Relation Extraction for Combination Drug Therapies. Please down the zipped file of this repo and follow the below instructions with the downloaded folder as the root directory. 
## Get access to the dataset
```
git clone https://github.com/allenai/drug-combo-extraction.git
```
Please see the paper [A Dataset for N-ary Relation Extraction of Drug Combinations](https://arxiv.org/abs/2205.02289) for more information about CombDrugExt dataset.
### Preprocess dataset to our expected format
```
python preprocess.py
```
## Installing Seq2rel using Poetry
```
curl -sSL https://install.python-poetry.org | python3 -
cd seq2rel
poetry install
cd ..
```
## Training
Use the `allennlp train` command with `.jsonnet` config file we provid to train the model
```
train_data_path="drug-combo-extraction/n-ary/train.txt" \
valid_data_path="drug-combo-extraction/n-ary/valid.txt" \
dataset_size=1362 \
allennlp train "training-config-seq2rel/n-ary.jsonnet" \
    --serialization-dir "output"\
    --include-package "seq2rel" 
```
## Evaluation
Example for evaluation on `n-ary/test.txt` file with fine-tuned model `model.tar.gz` with positive combination f1 score
```
python evaluation.py\
          --model output/model.tar.gz\
          --test_file drug-combo-extraction/n-ary/test.txt\
          --metric positive_combination_f1\
