# On the Effectiveness of Code Representation in Automated Patch Correctness Assessment

## project structure

```bash
├── core
│   ├── ods: run ODS on our dataset
│   ├── representation: construct code representation

├── tool: tools used to extract code representation
│   ├── add.jar
│   ├── coming_jar
│   ├── jdiff.jar
│   ├── property_graph

├── bases

├── pymodels
│   ├──classification: models used in APCA tasks
│      ├── ensemble: model for fusion-based APCA
│      ├── feature & tfidf: models for feature-based APCA
│      ├── graph: models for graph-based APCA
│      ├── sequence: models for sequence-based APCA
│      ├── tree: models for tree-based APCA

├── configs: model config & task config

├── evaluation: evaluation metrics used in APCA tasks

├── factory: factory mode

├── tasks: tasks customized for APCA tasks

├── tbcnn: run TBCNN on our dataset

├── tokenizer: tokenizers used in APCA tasks

├── trainer: trainers used in APCA tasks

├── dataset
│   ├── custom.py: dataset customized for APCA tasks

├── data: patch files, intermediate result & dataset for different code representation
│   ├── patch.zip: original patch files in .patch format, including 2277 deduplicated patches,3 exclude due to representation extraction failure, 2274 usable

├── utils

├── train.py

├── test.py

├── pretrain.py: code for pre-train models

├── README.MD
```

## prerequisite

We used several tool to extract different types of code representation, please download and install them first.

```bash

# install joern

Please refer to https://github.com/joernio/joern to  install joern

# install coming

Please refer to https://github.com/SpoonLabs/coming to install coming

# install add

add.jar is included in ./tool

# install property-graph

Please refer to https://github.com/Zanbrachrissik/PropertyGraph to install property-graph
```


## extract code representation

go to corresponding folder

```bash
cd ./core/representation
```

extract feature representation

```bash
python graph_tree_feature.py --ttype="feature"
```

extract tree representation

```bash
python graph_tree_feature.py --ttype="ast"
```

extract graph representation

```bash
python graph_tree_feature.py --ttype="graph"
```

extract sequence representation

```bash
python sequence.py 
```

You can find feature extracting result under ./output


## dataset split
We employed a 5-fold cross validation in our experiment, please run dataset_split.py to split the dataset 

## train
In the yml formatted configuration file, please specify parameters such as the training set, validation set, model, etc. Please specify the configuration file when running the training script.

```bash
# feature,sequence,tree,graph representation
python train.py --rtype='single' --config_file='train_config.yml'

# fusion representation
python train.py --rtype='fusion'
```


## test
```bash
python test.py --config_file='test_config.yml'
```


