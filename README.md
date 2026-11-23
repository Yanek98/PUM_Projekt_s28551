# Rekomendacje gier na Steamie

Projekt PUM, etap 2, branch `zajecia-2`. Pierwszy DAG przygotowuje dane o grach, a drugi porównuje modele gatunków i preferencji graczy. Wspólnym kluczem jest `app_id`.

## Etap 2 · eksperymenty AutoML

W [zweryfikowanym przebiegu Airflow](reports/airflow_automl_run.json) ukończono pięć zadań i porównano 12 modeli. Random Forest osiągnął ważony F1 0,381 na 18 540 grach z katalogu. ShuffleNetV2 z głowicą MLP uzyskał mikro-F1 0,564 na 1 471 grach ze zrzutami ekranu. Na 9 557 674 późniejszych recenzjach boosting osiągnął ROC-AUC 0,691, a model oparty na samym historycznym odsetku pozytywnych ocen gry uzyskał 0,626.

Wyniki do oceny zawiera [automatyczny ranking TOP-3](reports/models_results.md). [Pełna lista prób](reports/automl_runs.json) obejmuje wszystkich kandydatów, metryki, parametry i identyfikatory MLflow. Aby obejrzeć zapisane eksperymenty po świeżym sklonowaniu projektu, rozpakuj trzy [archiwa MLflow](reports/mlflow) do `mlruns/`, zachowując numeryczne katalogi eksperymentów, i uruchom Docker Compose. Możesz również samodzielnie powtórzyć trening.

Po aktualizacji wszystkich trzech artefaktów przez DAG pierwszego etapu Airflow automatycznie uruchamia DAG `automl`. Aby wykorzystać już przygotowane dane, zbuduj środowisko i wyzwól trening ręcznie:

```sh
docker compose build airflow
docker compose up -d
docker compose exec airflow airflow dags trigger automl
docker compose exec airflow airflow dags list-runs -d automl
```

Trening działa na CPU. Obraz Dockera zawiera dwa zestawy wag ImageNet. Zadanie `check_inputs` sprawdza artefakty pierwszego etapu. Zadania treningowe zapisują osobny run MLflow dla każdego kandydata, a zadanie `publish` buduje ranking z ukończonych prób bieżącego przebiegu. [Airflow](http://localhost:8080/dags/automl/grid) pokazuje stan zadań. [MLflow](http://localhost:5000) udostępnia eksperymenty `Steam-tabular`, `Steam-images` i `Steam-big`. Wagi, klasyfikatory oraz parametry przetwarzania są w zakładce Artifacts.

| Zbiór | Zadanie i kandydaci | Walidacja i selekcja |
|---|---|---|
| Katalog | Predykcja głównego gatunku z tytułu, roku premiery, liczebności cech i języków. Regresja logistyczna, Random Forest, Extra Trees i MLP korzystają ze wspólnego TF-IDF/SVD dopasowanego do gier treningowych. | Podział gier z etapu 1. Ranking według ważonego F1. Dodatkowo makro-F1 i accuracy. |
| Obrazy | Wieloetykietowa predykcja gatunków. Zamrożone MobileNetV3-Small i ShuffleNetV2 ×0,5 przetwarzają zrzuty w rozmiarze 96 × 160. Dla każdej sieci trenujemy głowicę liniową i MLP przez 7 epok. Każdy checkpoint zawiera wagi sieci i głowicy, statystyki normalizacji oraz kolejność etykiet. | Oryginały i augmentacje w treningu. W walidacji uśredniamy prawdopodobieństwa dwóch oryginalnych zrzutów danej gry. Wybór według mikro-F1 z progiem 0,5. |
| Recenzje | Predykcja, czy gracz poleci grę. Cztery modele wykorzystują historię gier i użytkowników: popularność gry, regresję logistyczną, drzewo i boosting histogramowy. | Opinie do końca 2020 r. tworzą historię. Dane z 2021 r. służą do dopasowania klasyfikatorów, a dane z 2022 r. do walidacji. Ranking według ROC-AUC na wszystkich recenzjach walidacyjnych. |

**Przetwarzanie recenzji.** DuckDB wylicza liczbę i odsetek pozytywnych opinii historycznych osobno dla gry i gracza. Dla graczy z mniej niż dwiema opiniami i nieznanych gier używamy średniej globalnej. Wygładzanie odpowiada ośmiu obserwacjom z tą średnią. Odsetki dzielimy na 20 przedziałów, a liczebność historii na sześć grup. Dask zlicza wiersze o tych samych cechach i etykiecie w kolejnych partiach. Liczności stanowią wagi w treningu i obliczaniu metryk. Model można więc zastosować przed wystawieniem recenzji. Tabele potrzebne do obliczania cech są artefaktami wspólnego runu `historical-features`, wskazanego przez `lookup_run_id`.

## Dane

Ścieżki w tabeli są względne wobec folderu z rozpakowanymi zbiorami. W aktualnym workspace jest to `projekt-polaczony/Projekt_1_wybror-datasetow/raw_data`. Na innym komputerze można ustawić `STEAM_RAW_DIR` w pliku `.env` obok `compose.yaml`, np. `STEAM_RAW_DIR=./raw_data`.

| Zbiór | Pliki wejściowe | Zastosowanie |
|---|---|---|
| [Gaming Profiles 2025](https://www.kaggle.com/datasets/artyomkruglov/gaming-profiles-2025-steam-playstation-xbox) | `gaming_profiles/steam/games.csv` | Katalog gier, 98 248 wierszy i 7 kolumn. Identyfikator źródłowy to `gameid`. |
| [Steam Games Metadata ContentRecsys](https://www.kaggle.com/datasets/dulanmihimansa2004/full-screenshot-dataset-2) | `games_metadata/full-text-meta-dataset-clean.csv` i katalog `games_metadata/full-2-screenshots-dataset-clean/` | 14 704 obrazy JPEG z 7 352 gier. Nazwa obrazu zawiera `appid`, po którym dołączane są gatunki gry. |
| [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) | `games_recomendations/recommendations.csv` | 41 154 794 recenzje. `user_id`, `app_id` i `is_recommended` opisują preferencję gracza. |

Źródła, wymagane kolumny i progi jakości są w [include/config](include/config).

## Uruchomienie w Dockerze

W katalogu z `compose.yaml`, po rozpakowaniu danych do powyższych lokalizacji:

```sh
docker compose up -d --build
docker compose exec airflow airflow dags list
docker compose exec airflow airflow dags trigger ingest_eda_prep
docker compose exec airflow airflow dags list-runs -d ingest_eda_prep
```

Polecenie `trigger` wykonaj po pojawieniu się DAG-u na liście. Stan `success` oznacza ukończenie wszystkich 12 zadań. Graf i logi można obejrzeć w [lokalnym Airflow](http://localhost:8080). Użytkownik samodzielnej instancji to `admin`. Hasło generowane lokalnie można odczytać przez `docker compose exec airflow cat /opt/airflow/standalone_admin_password.txt`. Aby zakończyć pracę, uruchom `docker compose stop`. Historia wykonań pozostaje w woluminie Dockera.

## Co wykonuje DAG

W każdej ścieżce zadanie `ingest` czyta pliki lokalne, sprawdza nagłówek i zapisuje dane etapowe w Parquet. Zadanie `quality` uruchamia Great Expectations na metrykach wyliczonych z pełnego wejścia. Każdy warunek ma w logu wartość zaobserwowaną, próg i wynik. Niespełniony warunek przerywa daną ścieżkę. Następnie `eda` tworzy profil HTML w ydata-profiling, a `prep` zapisuje dane dla kolejnego etapu.

| Ścieżka | Przygotowanie | Wynik lokalny |
|---|---|---|
| Katalog | Normalizacja identyfikatora. Odrzucenie gier bez etykiety gatunku, uzupełnienie pustych pól opisowych, kodowanie języków i standaryzacja roku premiery oraz liczebności cech. Skaler dopasowany na części treningowej. Etykietą do klasyfikacji jest pierwszy gatunek z listy źródłowej. | `data/processed/tabular/clean.parquet` |
| Obrazy | Dekodowanie każdego JPEG i dopasowanie metadanych. Losowy podział gier 80/20 z ziarnem 42. Oryginały obu części oraz przyciemniona kopia każdego obrazu treningowego (jasność × 0,85) trafiają do jednego archiwum. | `data/processed/images/dataset.zip` zawiera obrazy i `manifest.csv` z identyfikatorem gry, ścieżką w ZIP, gatunkami, podziałem oraz wariantem. |
| Recenzje | DuckDB czyta CSV strumieniowo. Do EDA wybiera 10 000 wierszy według hasza identyfikatora recenzji. Podczas przygotowania odrzuca czasy gry powyżej 99,5. percentyla, zachowuje negatywne i pozytywne oceny i dodaje `log_hours`. | `data/processed/big/reviews.parquet`, kompresja Zstandard i grupy wierszy do odczytu partiami. |

## Wyniki etapu 1

W przebiegu `etap_1_lokalnie` wszystkie 12 zadań osiągnęło stan `success`. Stan i czasy wykonania poszczególnych zadań zapisano w [wyniku Airflow](reports/airflow_run.json). Wszystkie 24 kontrole jakości zakończyły się powodzeniem.

W katalogu 5 549 gier nie miało przypisanego gatunku. Po ich odfiltrowaniu do klasyfikacji zostało 92 699 gier. W metadanych obrazów 261 list gatunków miało brakujący nawias zamykający. Ingest uzupełnia ten znak i zachowuje wszystkie dostępne etykiety. Wśród recenzji próg 99,5. percentyla czasu gry wyniósł 928,9 godziny. Po odfiltrowaniu 205 471 obserwacji powyżej progu zostały 40 949 323 recenzje.

| Zbiór | Raport EDA | Kontrole Great Expectations | Liczebności po przygotowaniu |
|---|---|---|---|
| Katalog | [HTML](reports/tabular_eda.html) | [JSON](reports/tabular_quality.json) | [JSON](reports/tabular_prep.json) |
| Obrazy | [HTML](reports/images_eda.html) | [JSON](reports/images_quality.json) | [JSON](reports/images_prep.json) |
| Recenzje | [HTML](reports/big_eda.html) | [JSON](reports/big_quality.json) | [JSON](reports/big_prep.json) |

Raporty HTML można otworzyć lokalnie w przeglądarce. Pliki źródłowe i duże dane wynikowe są przechowywane lokalnie. Do repozytorium trafiają kod, konfiguracja i raporty.
