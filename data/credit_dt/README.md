# credit_dt dataset (not in Git)

`fraudTrain.csv` (~335 MB) and `fraudTest.csv` (~143 MB) exceed GitHub's file size limit and are listed in `.gitignore`.

## Obtain the files

Download from the original source (e.g. Kaggle *Credit Card Fraud Detection* / similar `credit_dt` release) and place:

```text
data/credit_dt/fraudTrain.csv
data/credit_dt/fraudTest.csv
```

## Verify

```bash
python scripts/check_setup.py
python main.py              # train (uses row caps in config.yaml)
python main.py --verify
```

Row caps in `config.yaml` (`max_train_rows`, `max_test_rows`) keep training practical on a laptop.
