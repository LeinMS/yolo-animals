import argparse
from huggingface_hub import snapshot_download
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF dataset repo, e.g. username/zapovednik_combined")
    ap.add_argument("--out", required=True, help="Output directory, e.g. /workspace/data")
    ap.add_argument("--revision", default=None, help="Optional HF revision/branch/tag")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        revision=args.revision,
    )
    print(f"Downloaded dataset to: {out_dir}")

if __name__ == "__main__":
    main()
