"""
====================================================
Demonstrate pre-specifying a dictionary for splitter
====================================================

This example shows how projection matrices are generated for a manifold tree,
specifically the :class:`sktree.tree.DictionaryDecisionTreeClassifier`.

In this method, the user may pre-specify a dictionary of possible filters to apply
onto the data. Generally, the dictionary should be highly sampled from the
desired space.

For instance, in this example, we will demonstrate how to specify a dictionary
that is sampled from the space of Gabor filters. We will utilize skimage to
generate the filters within the

Pre-specifying the possible projection matrices as a dictionary is useful
because this will alleviate the tree having to generate candidate projections
on the fly within each split node. This enables very quick testing of ideas.

Once a dictionary is deemed useful, one can then implement an on-the-fly sampling
of the projection vectors based on some space of filters (e.g. Gabor).
"""

# from skimage.feature import gaborfilter
# from scipy
import pywt

# %%
# Sample gabor filters and visualize them
# ---------------------------------------
# Using scikit-image, we can easily sample Gabor filters.
# For more information, see their documentation.
wavelet = pywt.ContinuousWavelet("morl")

# sample wavelets

# %%
# Pass these into a splitter and sample split candidates
# ------------------------------------------------------

# %%
# Discussion
# ----------
# As we see, the Gabor filters are pre-specified, and when we sample
# a projection vector within the splitter, it will sample a random filter
# among the candidates (i.e. columns of the dictionary).
#
# To use this splitter in practice, one can use the :class:`sktree.tree.DictionaryDecisionTreeClassifier`.
