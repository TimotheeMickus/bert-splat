# Github repo for "How To Dissect a Muppet"

This is the repository for "How to Dissect a Muppet: The Structure of
Transformer Embedding Spaces", [in TACL](https://aclanthology.org/2022.tacl-1.57/).

**Note:** the code here is not user-friendly. You're welcome to rework on it on
your own, we're only distributing it for future reference and reproductibility.

## How is this repository structured?
The main bit is to be found in `linear_structure.py`;  that's where we decompose
embeddings in sub-terms and define some other useful tools.

The remaining scripts correspond to different (sub-)experiments.


## References
If you found the paper or this repository useful, don't hesitate to cite our
paper:
```
@article{mickus-etal-2022-dissect,
    title = "How to Dissect a {M}uppet: The Structure of Transformer Embedding Spaces",
    author = "Mickus, Timothee  and
      Paperno, Denis  and
      Constant, Mathieu",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "10",
    year = "2022",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2022.tacl-1.57",
    doi = "10.1162/tacl_a_00501",
    pages = "981--996",
    abstract = "Pretrained embeddings based on the Transformer architecture have taken the NLP community by storm. We show that they can mathematically be reframed as a sum of vector factors and showcase how to use this reframing to study the impact of each component. We provide evidence that multi-head attentions and feed-forwards are not equally useful in all downstream applications, as well as a quantitative overview of the effects of finetuning on the overall embedding space. This approach allows us to draw connections to a wide range of previous studies, from vector space anisotropy to attention weights.",
}

```
For all intents and purposes, you can consider this repository licensed under
CC-BY-SA.
