import numpy as np
import pytest
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

from treeple.stats import NeuroExplainableOptimalFIT

"""
Experiments
"""
"""Use MNIST 3/5 Dataset"""
# load data
transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
indices = np.where((mnist.targets == 3) | (mnist.targets == 5))[0]
filtered_mnist = torch.utils.data.Subset(mnist, indices)
X_mnist = []
y_mnist = []
for image, label in filtered_mnist:
    X_mnist.append(image.numpy().flatten())
    y_mnist.append(label)
X_mnist = np.array(X_mnist)
y_mnist = np.array(y_mnist)
X_train_mnist, X_temp_mnist, y_train_mnist, y_temp_mnist = train_test_split(
    X_mnist, y_mnist, test_size=0.4, random_state=0
)
X_val_mnist, X_test_mnist, y_val_mnist, y_test_mnist = train_test_split(
    X_temp_mnist, y_temp_mnist, test_size=0.5, random_state=0
)

neofit = NeuroExplainableOptimalFIT(n_estimators=5000, n_permutations=100000, clf_type="SPORF")
# p_values = neofit.feat_imp_test(X_train_mnist, y_train_mnist)
p_values, imp_features, _ = neofit.get_significant_features(X_train_mnist, y_train_mnist)

X_val_mnist_ = X_val_mnist[:, imp_features]
X_test_mnist_ = X_test_mnist[:, imp_features]
clf_imp = RandomForestClassifier(n_estimators=500, random_state=0).fit(X_val_mnist_, y_val_mnist)
acc_fit_imp = clf_imp.score(X_test_mnist_, y_test_mnist)
clf_all = RandomForestClassifier(n_estimators=500, random_state=0).fit(X_val_mnist, y_val_mnist)
acc_fit_all = clf_all.score(X_test_mnist, y_test_mnist)

print(acc_fit_all, acc_fit_imp)


# TODO: feature importance visualization - heatmap
# TODO: experiments on trunk simulatioin
# def show_images(dataset, idx):
#     fig, ax = plt.subplots(figsize=(5, 5))
#     image, label = dataset[idx]
#     ax.imshow(image.squeeze(), cmap='gray')
#     ax.set_title(f'Label: {label}')
#     ax.axis('off')
#     plt.tight_layout()
#     plt.show()
#     return fig, ax

# # Display images
# show_images(filtered_mnist,1)


"""
Tests
"""


# Test errors
def test_neofit_errors():
    """Test classifier type error when training the explainer"""
    with pytest.raises(ValueError, match="Classifier type RF not implemented yet."):
        neofit = NeuroExplainableOptimalFIT(n_estimators=100, n_permutations=1000, clf_type="RF")
        neofit.construct_orf()


def test_neofit_params():
    """Test parameters settings"""
    neofit = NeuroExplainableOptimalFIT(n_estimators=100, n_permutations=5000, clf_type="SPORF")
    assert neofit.n_estimators == 100
    assert neofit.n_permutations == 5000
    # modify
    neofit.n_estimators = 1000
    neofit.n_permutations = 10000
    assert neofit.n_estimators == 1000
    assert neofit.n_permutations == 10000


if __name__ == "__main__":
    test_neofit_errors()
    test_neofit_params()
