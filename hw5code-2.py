import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices].astype(int)  # Приведение к целым числам

    unique_values = np.unique(sorted_features)
    if len(unique_values) == 1:
        return [], [], None, None

    thresholds = (unique_values[:-1] + unique_values[1:]) / 2

    ginis = []
    gini_best = None
    threshold_best = None

    for threshold in thresholds:
        left_mask = sorted_features < threshold
        right_mask = ~left_mask

        left_targets = sorted_targets[left_mask]
        right_targets = sorted_targets[right_mask]

        left_size = len(left_targets)
        right_size = len(right_targets)
        total_size = len(sorted_targets)

        if left_size == 0 or right_size == 0:
            continue

        left_probs = np.bincount(left_targets, minlength=2) / left_size
        right_probs = np.bincount(right_targets, minlength=2) / right_size

        left_gini = 1 - np.sum(left_probs ** 2)
        right_gini = 1 - np.sum(right_probs ** 2)

        gini = -(left_size / total_size) * left_gini - (right_size / total_size) * right_gini
        ginis.append(gini)

        if gini_best is None or gini > gini_best:
            gini_best = gini
            threshold_best = threshold

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if len(sub_y) <= self._min_samples_leaf or \
           (self._max_depth is not None and depth >= self._max_depth) or \
           np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_count / (current_click + 1e-6)
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini is not None and (gini_best is None or gini < gini_best):
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold
                
                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                  filter(lambda x: x[1] < threshold, categories_map.items())))
                    node["categories_split"] = set(threshold_best) if threshold_best else set(categories_map.keys())
                else:
                    raise ValueError

        if feature_best is not None and self._feature_types[feature_best] == "categorical" and not threshold_best:
            threshold_best = list(categories_map.keys())

        if feature_best is None or np.sum(split) < self._min_samples_leaf or np.sum(~split) < self._min_samples_leaf:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        node["threshold"] = threshold_best

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_split = node["feature_split"]
        if self._feature_types[feature_split] == "real":
            if x[feature_split] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif self._feature_types[feature_split] == "categorical":
            if "categories_split" in node and x[feature_split] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in X])

class LinearRegressionTree(DecisionTree):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None, n_quantiles=10):
        super().__init__(feature_types, max_depth, min_samples_split, min_samples_leaf)
        self.n_quantiles = n_quantiles

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if len(sub_y) <= 1 or (self._max_depth is not None and depth >= self._max_depth):
            node["type"] = "terminal"
            node["model"] = LinearRegression().fit(sub_X, sub_y)
            return

        feature_best, threshold_best, loss_best, split = None, None, None, None

        for feature in range(sub_X.shape[1]):
            feature_vector = sub_X[:, feature]

            thresholds = np.quantile(feature_vector, np.linspace(0, 1, self.n_quantiles + 2)[1:-1])

            for threshold in thresholds:
                left_mask = feature_vector < threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self._min_samples_leaf or np.sum(right_mask) < self._min_samples_leaf:
                    continue

                left_model = LinearRegression().fit(sub_X[left_mask], sub_y[left_mask])
                right_model = LinearRegression().fit(sub_X[right_mask], sub_y[right_mask])

                left_loss = mean_squared_error(sub_y[left_mask], left_model.predict(sub_X[left_mask]))
                right_loss = mean_squared_error(sub_y[right_mask], right_model.predict(sub_X[right_mask]))

                total_loss = (np.sum(left_mask) / len(sub_y)) * left_loss + (np.sum(right_mask) / len(sub_y)) * right_loss

                if loss_best is None or total_loss < loss_best:
                    feature_best = feature
                    threshold_best = threshold
                    loss_best = total_loss
                    split = left_mask

        if feature_best is None:
            node["type"] = "terminal"
            node["model"] = LinearRegression().fit(sub_X, sub_y)
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        node["threshold"] = threshold_best

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["model"].predict([x])[0]

        if x[node["feature_split"]] < node["threshold"]:
            return self._predict_node(x, node["left_child"])
        else:
            return self._predict_node(x, node["right_child"])

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in X])

