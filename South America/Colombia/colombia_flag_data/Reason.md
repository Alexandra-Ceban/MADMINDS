

reviews.json` contains 1,000 trusted reference reviews and 1,200 indexed
samples. I used the trusted reviews as a text baseline and looked for unusual
Unicode characters in the samples. This found 12 candidate records: six with
zero-width spaces and six with Cyrillic letters that resemble Latin letters.
Their `sample_index` values are listed by `solve_reviews.py.

The archive note says to hash the comma-separated candidate indices with
SHA-256, XOR that 32-byte result with each `archive_tag`, and remove trailing
zero bytes. Applying that procedure to these 12 candidates did **not** produce
a `SIGMOID_` message, so this is not yet a verified complete set of altered
records. The script deliberately reports that result instead of presenting an
unverified flag.
