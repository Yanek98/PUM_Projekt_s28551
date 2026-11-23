# Steam · wyniki AutoML

Przebieg Airflow `etap_2_steam`. Wyniki na walidacji.

## Katalog gier

| Miejsce | Model | f1_weighted | MLflow run ID |
|---:|---|---:|---|
| 1 | RandomForest | 0.380999 | `e35e76340e9140d391e467454efe969b` |
| 2 | MLP | 0.370831 | `dbcf90a1a6f84132929acdd85f685ac1` |
| 3 | LogisticRegression | 0.332033 | `ccae762e840b4f0e997339c37cdd6d54` |

## Zrzuty ekranu

| Miejsce | Model | f1_micro | MLflow run ID |
|---:|---|---:|---|
| 1 | ShuffleNetV2-MLP | 0.563787 | `a1082d942da443abbf4d488f510a7e72` |
| 2 | MobileNetV3Small-MLP | 0.562857 | `13980dc44e63480eb656251feb55d7a1` |
| 3 | ShuffleNetV2-Linear | 0.558945 | `59728881fed748e98c5d8a8640fb4e50` |

## Recenzje

| Miejsce | Model | roc_auc | MLflow run ID |
|---:|---|---:|---|
| 1 | HistogramBoost | 0.690604 | `5ba8297ceadf4e6eb576e1f143d9d5d7` |
| 2 | WeightedTree | 0.688120 | `99b6abfe23c3414a9c908545ddd1f57b` |
| 3 | WeightedLogistic | 0.686149 | `8e13165f79274fb58549bcdc77a4ec21` |

Wszystkie porównane modele, parametry i metryki zawiera [zestawienie JSON](automl_runs.json).

Archiwa MLflow z wagami i metrykami: [katalog](mlflow/tabular.zip), [obrazy](mlflow/images.zip), [recenzje](mlflow/big.zip).
