# Training a Network

---
The datasets in 200 pile-up events are quite large with approximately
270k hits per event.
While earlier we created and used data sets that only have pile-up applied
in a small region to keep the events down to a manageable size of
approximately 60k hits, we now perform the training in two steps:

1. Pre-clustering step
  This step uses the 'snowflake' approach to condense the dataset by ~97%.
2. Pre-clustered-to-End step
  Taking the condensed data we now can train the end-to-end step without
  running into memory constraints.

## Train Pre-clustering

Use model in `Train/pre_snowflake.py` on full 200 pile-up data.

```bash
python3 Train/pre_snowflake.py /path/to/data /output/path
```

## Train Pre-clustered-to-End

Use model in `Train/full_model_preclustered.py` on data that has been
pre-clustered with the snowflake model.

```bash
python3 Train/full_model_preclustered.py /path/to/preclustered/ /output/path
```
