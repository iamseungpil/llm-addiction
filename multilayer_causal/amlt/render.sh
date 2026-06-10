#!/bin/bash
# Render amlt yamls: substitute the local cached HF token into *.yaml.template.
# Rendered *.rendered.yaml files are gitignored — never commit real tokens.
set -euo pipefail
cd "$(dirname "$0")"
TOK=$(python -c "from huggingface_hub import get_token; print(get_token())")
for t in *.yaml.template; do
  out="${t%.yaml.template}.rendered.yaml"
  sed "s/HF_TOKEN_PLACEHOLDER/$TOK/g" "$t" > "$out"
  echo "rendered $out"
done
