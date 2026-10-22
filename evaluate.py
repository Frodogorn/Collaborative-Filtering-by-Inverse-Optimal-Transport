
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score a completed ratings table on held-out ratings.')
    parser.add_argument("--pred", type=str, default="output.npy",
                      help="Completed table produced by generate.py")
    parser.add_argument("--truth", type=str, default="data/ratings_test.npy",
                      help="Held-out ratings (NaN = not observed)")

    args = parser.parse_args()



    pred = np.load(args.pred)
    truth = np.load(args.truth)
    assert pred.shape == truth.shape, f"shape mismatch: {pred.shape} vs {truth.shape}"

    mask = ~np.isnan(truth)
    assert not np.isnan(pred[mask]).any(), "prediction contains NaN on evaluated cells"

    err = pred[mask] - truth[mask]
    rmse = np.sqrt(np.mean(err ** 2))

    # Exact-rating accuracy: prediction rounded to the nearest valid rating (0.5 steps in [0.5, 5])
    rounded = np.clip(np.round(pred[mask] * 2) / 2, 0.5, 5)
    accuracy = np.mean(rounded == truth[mask])

    print(f"Evaluated ratings: {mask.sum()}")
    print(f"RMSE:     {rmse:.4f}")
    print(f"Accuracy: {accuracy:.2%}")
