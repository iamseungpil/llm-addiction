#!/bin/bash
# Render amlt yamls: substitute the local cached HF token into *.yaml.template.
# Rendered *.rendered.yaml files are gitignored — never commit real tokens.
set -euo pipefail
cd "$(dirname "$0")"
# System python lacks huggingface_hub; use the amlt env python which also reads
# the wandb key from ~/.netrc.
PY=~/miniconda3/envs/amlt/bin/python
TOK=$($PY -c "from huggingface_hub import get_token; print(get_token())")
WANDB=$($PY -c "import netrc; a=netrc.netrc().authenticators('api.wandb.ai'); print(a[2] if a else '')")
for t in *.yaml.template; do
  out="${t%.yaml.template}.rendered.yaml"
  sed -e "s/HF_TOKEN_PLACEHOLDER/$TOK/g" \
      -e "s|WANDB_API_KEY_PLACEHOLDER|$WANDB|g" "$t" > "$out"
  echo "rendered $out"
done
