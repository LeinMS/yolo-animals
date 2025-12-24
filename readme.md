# YOLO remote training (Vast.ai)

## 1) Setup (inside Vast instance)
```bash
sudo apt-get update
sudo apt-get install -y python3-venv git

git clone <YOUR_GITHUB_REPO_URL>
cd yolo-remote-train

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
