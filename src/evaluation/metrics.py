
def compute_ap(retrieved_labels, query_label, k=10):
    """Computes Average Precision at K for a query label and retrieved labels."""
    ap = 0.0
    correct_count = 0
    for i in range(min(len(retrieved_labels), k)):
        if retrieved_labels[i] == query_label:
            correct_count += 1
            precision_at_i = correct_count / (i + 1)
            ap += precision_at_i
    if correct_count > 0:
        ap /= correct_count
    return ap


def compute_rr(retrieved_labels, query_label, k=10):
    """Computes Reciprocal Rank at K for a query label and retrieved labels."""
    for rank_idx, label in enumerate(retrieved_labels[:k]):
        if label == query_label:
            return 1.0 / (rank_idx + 1)
    return 0.0


def compute_precision_at_k(retrieved_labels, query_label, k):
    """Computes Precision at K for a query label and retrieved labels."""
    if not retrieved_labels or k <= 0:
        return 0.0
    sub_list = retrieved_labels[:k]
    matches = sum(1.0 for label in sub_list if label == query_label)
    return matches / k
