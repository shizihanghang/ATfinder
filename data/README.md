# Data

`dataset.csv` is the only sequence/label table. Split files store indices into this table. Sequences are not duplicated across folds.

## dataset.csv

| column | description |
|---|---|
| global_idx | row id, equal to 0..n-1 |
| sequence | peptide sequence |
| Breast ... Stomach | observed labels; 1 = observed positive, 0 = unlabeled |

There are 1953 peptides and 12 cancer-type labels. Unlabeled entries are not confirmed negatives.

## splits.json

- `independent`: 195 held-out Independent indices
- `folds`: five CV folds, each with `train` and `test` indices
- Independent samples are disjoint from all CV indices

## masks/

These files keep the frozen per-position loss/eval mask. Columns are `global_idx` plus the 12 labels. A value of 1 means the position is used; all observed positives are included.

- `fold_k_train.csv`
- `fold_k_test.csv`
- `independent.csv`

## embeddings and physicochemical table

- `esm2_t30_150m_mean.pt`: frozen ESM-2 t30 150M mean embeddings, aligned to `dataset.csv` row order
- `aaindex.csv`: AAindex physicochemical features
