# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

FLOSS introduces training-free, label-free class-wise prompt template selection for open-vocabulary semantic segmentation. Its central idea is that each semantic class may be better segmented by a small subset of individual CLIP prompt templates than by the standard average over all templates.

## Approach

The method starts from a frozen CLIP-based OVSS model and the 80 standard ImageNet prompt templates. For each template, it builds a single-template text classifier over the target class names and runs it on unlabeled target-domain images. For each class, FLOSS computes a class-wise uncertainty proxy over pixels predicted as that class; templates with the lowest entropy are selected as class experts. The paper uses the top four templates per class in its main experiments.

After selection, FLOSS builds one expert classifier per class from that class's selected templates. Each expert produces a full soft segmentation map. At each pixel, the final prediction is taken from experts that predict their own class of expertise; if none do, the method falls back to the default averaged-template classifier. The visual encoder is shared, so the main added cost is the extra text-classifier similarity and fusion work.

## Main Claims

Primary claim: class-wise template selection from unlabeled images improves frozen CLIP-based OVSS segmentation quality without labels or training.

Secondary claims: the selected experts generalize to related shifted domains, the approach works with very few unlabeled selection images, it is complementary to several OVSS backbones, and entropy is an effective unsupervised proxy for selecting useful class experts.
