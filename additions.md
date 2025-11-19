## Parallel Execution of Preprocessing

uncomment these two lines in index_builder.py file

```Python
with Pool(cpu_count()) as p:
   tokenized_chunks = p.map(preprocess_for_bm25, all_chunks)
```

## Building Index with Metric Evaluation

To calculate the metrics (Precision@k, Recall@k, Mean Reciprocal Rank, F1 Score) while building index, add the `--score` flag after index building command.

**Example:**

```bash
make run-index --score
```
