import os
import numpy as np
import pandas as pd
from core.data.preprocessing import add_basic_indicators
from ai_predictive.patterns import windowed_patterns
from core.models.vectorstore import VectorIndex


def make_synthetic(n: int = 600) -> pd.DataFrame:
    np.random.seed(42)
    price = 100 + np.cumsum(np.random.randn(n))
    df = pd.DataFrame({"Close": price})
    df = add_basic_indicators(df)
    return df


def main():
    df = make_synthetic()
    feature_cols = [c for c in ["Close", "SMA_10", "SMA_50", "RET_1", "VOL_10"] if c in df.columns]
    X, meta = windowed_patterns(df, feature_cols=feature_cols, window=30, stride=5, normalize=True)
    if X.shape[0] == 0:
        print("Not enough data for patterns.")
        return

    dim = X.shape[1]
    index = VectorIndex(dim=dim, metric="l2")
    index.add(X, metadata=meta)

    q = X[-1]
    res = index.search(q, k=5)
    print("Top-5 nearest pattern windows:")
    for i in range(res.indices.shape[1]):
        md = res.metadata[0][i]
        dist = res.distances[0][i]
        print(f"  #{i+1}: indices=({md.get('start')}, {md.get('end')}), ts=({md.get('ts_start')}, {md.get('ts_end')}), dist={float(dist):.6f}")

    out_dir = os.path.join(".", "vector_index_demo")
    index.save(out_dir)
    loaded = VectorIndex.load(out_dir)
    res2 = loaded.search(q, k=3)
    print("Reloaded index query (top-3):")
    for i in range(res2.indices.shape[1]):
        md = res2.metadata[0][i]
        dist = res2.distances[0][i]
        print(f"  #{i+1}: indices=({md.get('start')}, {md.get('end')}), dist={float(dist):.6f}")


+if __name__ == "__main__":
+    main()
