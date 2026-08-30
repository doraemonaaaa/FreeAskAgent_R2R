#!/usr/bin/env bash
# Download a public HF repo with curl (hf_hub cannot use this proxy).
#   download_hf.sh <org/name> <dest_dir> [parallel]
set -u
repo=$1; dest=$2; par=${3:-3}
mkdir -p "$dest"; cd "$dest"
curl -s --max-time 30 "https://huggingface.co/api/models/$repo" | python3 -c "import sys,json; print('\n'.join(f['rfilename'] for f in json.load(sys.stdin)['siblings'] if not f['rfilename'].startswith('.git')))" > files.txt
xargs -P "$par" -I{} sh -c 'mkdir -p "$(dirname {})"; curl -sL --retry 8 --retry-delay 10 --retry-all-errors -C - -o "{}" "https://huggingface.co/'"$repo"'/resolve/main/{}"' < files.txt
# verify sizes against the hub
curl -s --max-time 30 "https://huggingface.co/api/models/$repo?blobs=true" | python3 -c "
import sys,json,os
d=json.load(sys.stdin); bad=[f['rfilename'] for f in d['siblings'] if not f['rfilename'].startswith('.git') and not (os.path.exists(f['rfilename']) and (f.get('size') is None or os.path.getsize(f['rfilename'])==f['size']))]
print('VERIFY', 'OK' if not bad else 'MISSING '+' '.join(bad))"
