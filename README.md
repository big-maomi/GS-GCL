# GS-GCL (General-Special Neighborhood-enhanced Contrastive Learning)

## Overview

We propose a contrastive learning paradigm, named General-Special Neighborhood-enrhanced Contrastive Learning (**GS-GCL**), to explicitly capture potential node relatedness into contrastive learning for graph collaborative filtering.

Yang Zezheng proposed this model


## Requirements

```
recbole==1.0.0
python==3.7.7
pytorch==1.7.1
faiss-gpu==1.7.1
cudatoolkit==10.1
```



## Quick Start

```bash
python main.py --dataset ml-1m
# When downloading the dataset for the first time, an error occurs, 
# but it resolves afterward, and re-entering the command works fine.
python main.py --dataset ml-1m
```

You can replace `ml-1m` to `yelp`, `amazon-books`, `gowalla-merged` to reproduce the results reported in our paper.



## Acnowledgement

The implementation is based on the open-source recommendation library [RecBole](https://github.com/RUCAIBox/RecBole).



