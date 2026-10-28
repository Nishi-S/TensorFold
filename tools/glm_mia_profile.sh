#!/bin/sh
# Explicit native GLM catch-up profile; override individual variables before invoking this wrapper.
set -eu
if [ "$#" -eq 0 ]; then
    echo 'Usage: tools/glm_mia_profile.sh COMMAND [ARGS...]' >&2
    exit 2
fi
export TF_GLM_DENSE_BITS="${TF_GLM_DENSE_BITS:-4}"
export TF_GLM_DENSE_FORMAT="${TF_GLM_DENSE_FORMAT:-affine4}"
export TF_GLM_HEAD_BITS="${TF_GLM_HEAD_BITS:-8}"
export TF_GLM_HEAD_FORMAT="${TF_GLM_HEAD_FORMAT:-fp8}"
export TF_EXL3_DECODE_LOADS="${TF_EXL3_DECODE_LOADS:-nc1}"
export TF_EXL3_DECODE_FUSE="${TF_EXL3_DECODE_FUSE:-0}"
export TF_GLM_L2PF="${TF_GLM_L2PF:-1}"
export TF_GLM_L2PF_MB="${TF_GLM_L2PF_MB:-8}"
export TF_GLM_L2PF_ROWS="${TF_GLM_L2PF_ROWS:-64}"
export TF_KV_PAGES="${TF_KV_PAGES:-0}"
exec "$@"
