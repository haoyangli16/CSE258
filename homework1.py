"""
CSE258 Homework 1

Basic approach:
- Q1-4: Linear regression with different temporal features (direct vs one-hot encoding)
- Q5-7: Logistic regression for beer review classification (positive/negative)

To run: python homework1.py --data_dir "path/to/datasets"
"""

import numpy
from sklearn import linear_model

# Q1-4: Regression on book reviews


def getMaxLen(dataset):
    maxLen = max(len(d["review_text"]) for d in dataset)
    # print(f"Max review length: {maxLen}")
    return maxLen


def featureQ1(datum, maxLen):
    length = len(datum["review_text"])
    return [1, length / maxLen]


def Q1(dataset):
    maxLen = getMaxLen(dataset)

    X = numpy.array([featureQ1(d, maxLen) for d in dataset])
    y = numpy.array([d["rating"] for d in dataset])
    # print(f"X shape: {X.shape}, y shape: {y.shape}")

    # fit_intercept=False since we're adding the intercept term manually
    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(X, y)

    theta = model.coef_
    predictions = model.predict(X)
    MSE = float(numpy.mean((predictions - y) ** 2))  # need float not numpy.float64

    return theta, MSE


def featureQ2(datum, maxLen):
    # One-hot for weekday and month
    # Drop Monday(0) and Jan(1) as reference
    length = len(datum["review_text"])
    feat = [1, length / maxLen]

    t = datum["parsed_date"]
    weekday = t.weekday()
    month = t.month

    # One-hot encode days Tue-Sun (skip Mon)
    for i in range(1, 7):
        feat.append(1 if weekday == i else 0)

    # One-hot encode months Feb-Dec (skip Jan)
    for i in range(2, 13):
        feat.append(1 if month == i else 0)

    # print(f"Feature vector length: {len(feat)}") 
    return feat


def Q2(dataset):
    maxLen = getMaxLen(dataset)

    X = numpy.array([featureQ2(d, maxLen) for d in dataset])
    y = numpy.array([d["rating"] for d in dataset])
    # print(f"First feature: {X[0]}") 

    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(X, y)

    predictions = model.predict(X)
    MSE = float(numpy.mean((predictions - y) ** 2))

    return X, y, MSE


def featureQ3(datum, maxLen):
    # Just use weekday/month directly as numbers instead of one-hot
    length = len(datum["review_text"])
    t = datum["parsed_date"]

    return [1, length / maxLen, t.weekday(), t.month]


def Q3(dataset):
    maxLen = getMaxLen(dataset)

    X = numpy.array([featureQ3(d, maxLen) for d in dataset])
    y = numpy.array([d["rating"] for d in dataset])

    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(X, y)

    predictions = model.predict(X)
    MSE = float(numpy.mean((predictions - y) ** 2))
    # print(f"Q3 MSE: {MSE}, Q2 should be similar or better")

    return X, y, MSE


def Q4(dataset):
    # Train/test split - need to see if the models overfit
    maxLen = getMaxLen(dataset)

    split = len(dataset) // 2
    train = dataset[:split]
    test = dataset[split:]
    # print(f"Train size: {len(train)}, Test size: {len(test)}")

    # Model 2 (one-hot)
    X_train2 = numpy.array([featureQ2(d, maxLen) for d in train])
    y_train = numpy.array([d["rating"] for d in train])

    model2 = linear_model.LinearRegression(fit_intercept=False)
    model2.fit(X_train2, y_train)

    X_test2 = numpy.array([featureQ2(d, maxLen) for d in test])
    y_test = numpy.array([d["rating"] for d in test])
    pred2 = model2.predict(X_test2)
    mse2 = float(numpy.mean((pred2 - y_test) ** 2))

    # Model 3 (direct encoding)
    X_train3 = numpy.array([featureQ3(d, maxLen) for d in train])
    model3 = linear_model.LinearRegression(fit_intercept=False)
    model3.fit(X_train3, y_train)

    X_test3 = numpy.array([featureQ3(d, maxLen) for d in test])
    pred3 = model3.predict(X_test3)
    mse3 = float(numpy.mean((pred3 - y_test) ** 2))

    return mse2, mse3


# Q5-7: Classification on beer reviews


def featureQ5(datum):
    return [1, len(datum["review/text"])]


def Q5(dataset, feat_func):
    # Binary classification: positive (>=4) vs negative (<4)
    X = numpy.array([feat_func(d) for d in dataset])
    y = numpy.array([1 if d["review/overall"] >= 4 else 0 for d in dataset])
    # print(f"Positive examples: {sum(y)}, Negative: {len(y) - sum(y)}") 

    # balanced weights so model doesn't just predict majority class
    model = linear_model.LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X, y)

    preds = model.predict(X)

    TP = int(numpy.sum((preds == 1) & (y == 1)))
    TN = int(numpy.sum((preds == 0) & (y == 0)))
    FP = int(numpy.sum((preds == 1) & (y == 0)))
    FN = int(numpy.sum((preds == 0) & (y == 1)))

    FPR = FP / (TN + FP) if (TN + FP) > 0 else 0
    FNR = FN / (TP + FN) if (TP + FN) > 0 else 0
    BER = float(0.5 * (FPR + FNR))

    return TP, TN, FP, FN, BER


def Q6(dataset):
    X = numpy.array([featureQ5(d) for d in dataset])
    y = numpy.array([1 if d["review/overall"] >= 4 else 0 for d in dataset])

    model = linear_model.LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X, y)

    probs = model.predict_proba(X)[:, 1]  # probability of positive class
    sorted_idx = numpy.argsort(-probs)  # sort descending
    # print(f"Top 5 probs: {probs[sorted_idx[:5]]}")

    precs = []
    for K in [1, 100, 1000, 10000]:
        top_k = sorted_idx[:K]
        precision = numpy.sum(y[top_k]) / K
        precs.append(precision)

    return precs


def featureQ7(datum):
    # Richer features to improve BER
    # Initially tried just text features (exclamation marks, word counts) but only got ~1% improvement
    # Adding the individual aspect ratings gave much better results
    feat = [1, len(datum["review/text"])]

    # The individual aspect scores are super predictive
    feat.append(datum.get("review/appearance", 2.5))
    feat.append(datum.get("review/aroma", 2.5))
    feat.append(datum.get("review/palate", 2.5))
    feat.append(datum.get("review/taste", 2.5))

    feat.append(datum.get("beer/ABV", 5.0))

    text = datum["review/text"]
    feat.append(text.count("!"))
    feat.append(text.count("great"))
    feat.append(text.count("bad"))
    feat.append(text.count("love"))
    # print(f"Feature count: {len(feat)}")

    return feat


# Local testing

if __name__ == "__main__":
    import argparse
    import gzip
    import json
    import random

    import dateutil.parser

    parser = argparse.ArgumentParser(description="CSE258 HW1")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/tommy/Projects/personal/courses/cse258/datasets",
    )
    args = parser.parse_args()

    # Load book reviews
    print("Loading fantasy dataset...")
    f = gzip.open(f"{args.data_dir}/fantasy_10000.json.gz")
    dataset = []
    for line in f:
        dataset.append(json.loads(line))
    f.close()
    # print(f"Loaded {len(dataset)} reviews")

    # Parse dates for temporal features
    for d in dataset:
        d["parsed_date"] = dateutil.parser.parse(d["date_added"])

    print("Testing Q1-Q4...")
    theta1, mse1 = Q1(dataset)
    print(f"Q1 - theta: {theta1}, MSE: {mse1:.4f}")

    X2, Y2, mse2 = Q2(dataset)
    print(f"Q2 - MSE: {mse2:.4f}, feat dim: {len(X2[0])}")

    X3, Y3, mse3 = Q3(dataset)
    print(f"Q3 - MSE: {mse3:.4f}")

    # Q4 needs shuffled data
    dataset_shuffled = dataset[:]
    random.seed(0)
    random.shuffle(dataset_shuffled)
    test_mse2, test_mse3 = Q4(dataset_shuffled)
    print(
        f"Q4 - Test MSE (one-hot): {test_mse2:.4f}, Test MSE (direct): {test_mse3:.4f}"
    )

    # Load beer reviews
    print("\nLoading beer dataset...")
    f = open(f"{args.data_dir}/beer_50000.json")
    beer_data = []
    for line in f:
        beer_data.append(eval(line))
    f.close()

    print("Testing Q5-Q7...")
    TP, TN, FP, FN, ber5 = Q5(beer_data, featureQ5)
    print(f"Q5 - TP:{TP} TN:{TN} FP:{FP} FN:{FN}, BER: {ber5:.4f}")

    precs = Q6(beer_data)
    print(
        f"Q6 - P@1:{precs[0]:.3f} P@100:{precs[1]:.3f} P@1K:{precs[2]:.3f} P@10K:{precs[3]:.3f}"
    )

    _, _, _, _, ber7 = Q5(beer_data, featureQ7)
    improvement = ber5 - ber7
    print(
        f"Q7 - BER: {ber7:.4f} (improved by {improvement:.4f} = {improvement / ber5 * 100:.1f}%)"
    )

    print("\nDone!")
