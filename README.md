# multiplex_gnn

This repository is the implementation of ([Multiplex Graph Neural Networks for Multi-behavior Recommendation](https://dl.acm.org/doi/abs/10.1145/3340531.3412119)). MGNN tackles the multi-behavior recommendation problem from a novel perspective, i.e., the perspective of link prediction in multiplex networks. By taking advantage of both the multiplex network structure and graph representation learning techniques, MGNN learns shared embeddings and behavior-specific embeddings for users and items to model the collective effect of multiple types of behaviors.

> Weifeng Zhang, Jingwen Mao, Yi Cao, and Congfu Xu. 2020. Multiplex Graph Neural Networks for Multi-behavior Recommendation. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management (CIKM '20). Association for Computing Machinery, New York, NY, USA, 2313–2316. DOI:https://doi.org/10.1145/3340531.3412119

### Run the code
  ```
  $ cd src
  $ python run.py --dataset steam (or yoochoose, or db_book)
  ```

The preprocessed datasets but not the original datasets are store in the directory `/data/`. You can also download the original datasets and run the preprocess code.

### Download datasets

- Yoochoose

[Download](https://www.kaggle.com/chadgostopp/recsys-challenge-2015) the dataset into the directory `/data/yoochoose/`.

- Steam

[Download](https://cseweb.ucsd.edu/%7Ejmcauley/datasets.html#steam_data) and unzip the Version 1 of the dataset into the directory `/data/steam/`.

- DB-book

[Download](https://github.com/7thsword/MFPR-Datasets/) the dataset into the directory `/data/db_book/`.

Run the corresponding preprocessing code in the 'preprocess.py' file.

