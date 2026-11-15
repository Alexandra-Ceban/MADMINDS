The flag is stored in the `embedding.weight` tensor inside `model.safetensors`, not as a text string. The tensor has shape `256 × 32` and type `F32`. Its row with index **128** is not used by any character in the supplied `corpus.csv`.

The first 18 float values in that row are small multiples of `1/1024`. To decode them, multiply each value by 1024, round to the nearest integer, add 128, and interpret the resulting bytes as ASCII. This produces:

SIGMOID_PROVENANCE

Because row 128 is outside the character IDs used by the corpus, the payload does not affect the classifier's observed predictions.

The extractor reads the Safetensors header and tensor byte offsets directly; it does not deserialize or execute model code. It also checks the tensor layout and verifies that token ID 128 is absent from the corpus text.
