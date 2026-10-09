# Data provenance

The source is Harry Wang's [Crypto Coven dataset](https://www.kaggle.com/datasets/harrywang/crypto-coven), collected on 21 April 2022 from OpenSea. The author's [Hugging Face mirror](https://huggingface.co/datasets/harrywang/crypto-coven) declares the MIT license. Author: Harry Wang. These data retain their source attribution; they are not newly collected by this project.

The included `witches_traits.csv` is a column-only extraction from `witches.csv`: six numeric traits, 26 categorical traits, token ID, last-sale amount, and last-sale timestamp. All 9,764 source rows are preserved. Owner addresses, names, descriptions, URLs, and cumulative sales counts are excluded. Tokens are deduplicated during preprocessing, not in the distributed file.

The pinned public mirror was downloaded and compared with the local coursework snapshot using pandas; every cell matched. Revision: `50b606e9d78a38848b8452d72072678cbeaedd05`. Run `python download_data.py` to obtain this raw source, then `python -m src.train --data data/witches.csv` to reproduce the same experiment from it. The helper preserves any existing file. The raw download is ignored by Git.

The original expanded OpenSea export was checked for payment currency and decimals: 4,520 priced rows use ETH and 376 use WETH, all with 18 decimal places. The amount is therefore divided by 10**18 and expressed as ETH-equivalent. The source card calls the raw amount "gwei", but its stated 18-digit scaling and the payment-token metadata imply wei. We do not use its USD price conversion.

This is a **latest-sale snapshot per NFT**, not a transaction history. The absence of a sale excludes a token from supervised learning. Splitting by last-sale date does not remove snapshot selection bias or reconstruct information available to a live investor. Raw source and full token-level predictions are not published here; the curated public traits and aggregate results are included.
