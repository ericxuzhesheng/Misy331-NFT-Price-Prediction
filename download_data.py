"""Download the original author's public dataset mirror; no account required."""
import argparse
from pathlib import Path
from urllib.request import urlopen

URL = 'https://huggingface.co/datasets/harrywang/crypto-coven/resolve/50b606e9d78a38848b8452d72072678cbeaedd05/witches.csv'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/witches.csv')
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise SystemExit(f'{path} already exists; choose another output to preserve existing data.')
    with urlopen(URL, timeout=120) as response:
        content = response.read()
    if not content.startswith(b'id,num_sales,'):
        raise SystemExit('Unexpected dataset response; download witches.csv from the source page.')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    print(f'Downloaded {len(content):,} bytes to {path}')
