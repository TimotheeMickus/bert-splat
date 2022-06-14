# Github repo for "How To Dissect a Muppet"

This is the repository for "How to Dissect a Muppet: The Structure of
Transformer Embedding Spaces", to appear in TACL. Here's an [arxiv 
preprint](https://arxiv.org/abs/2206.03529) for now.

**Note:** the code here is not user-friendly. You're welcome to rework on it on
your own, we're only distirbuting it for future reference and reproductibility.

## How is this repository structured?
The main bit is to be found in `linear_structure.py`;  that's where we decompose
embeddings in sub-terms and define some other useful tools.

The remaining scripts correspond to different (sub-)experiments.


## References
If you found the paper or this repository useful, don't hesitate to cite our
paper:
```
@misc{mickus-etal-2022-dissect-preprint,
  author = {Mickus, Timothee and Paperno, Denis and Constant, Mathieu},
  title = {How to Dissect a Muppet: The Structure of Transformer Embedding Spaces},
  publisher = {arXiv},
  year = {2022}
}

```
For all intents and purposes, you can consider this repository licensed under
CC-BY-SA.
