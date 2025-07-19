## MTL-CR

## Requirements

### Minimize requirements

The list of minimize requirements can be found in `requirements.txt`.

### Additional requirements

If you need to reprocess the raw dataset, or use your own dataset,
then you will also need to install the following packages.
```
tree_sitter==0.22.3
antlr4-python3-runtime==4.9.2
```
Besides, `antlr4` need to be installed,
[installation guidance here](https://github.com/antlr/antlr4/blob/master/doc/getting-started.md).

If you encounter errors about `my-languages.so` when preprocessing the dataset, 
please run `sources/data/asts/build_lib.py` first.

## Datasets and Tokenizers

We provide pre-processed datasets, saved as pickle binary files, 
which can be loaded directly as instances.

The pre-processed datasets can be downloaded here:
Put the downloaded dataset pickle file into `{dataset_root}/dataset_saved/` (default to`.../dataset/dataset_saved`), 
the program will automatically detect and use it.


##  Pre-trained Tokenizers and Models

Custom tokenizers (we call "vocab") can be downloaded here: 
Extract it in a certain directory. 
Specific the argument `trained_vocab` of `main.py` 
where the tokenizers are located or put it in `{dataset_root}/vocab_saved` (default to`.../dataset/vocab_saved`).


## Runs

Run `main.py` . 
All arguments are located in `args.py`, specific whatever you need.




