#!/bin/bash
# READ-ONLY scan of the pod disks (network volume /workspace + container disk /root).
# Lists what takes space and flags clean-up CANDIDATES. It deletes nothing; deletion is a separate,
# explicit step after the user has reviewed the list (rule: look at the target before deleting;
# never destroy completed-work outputs: eval results, training runs, scores).
#   bash r1_pod_scan.sh > /root/pod_scan.txt 2>&1
set -u
echo "=== quota / free ==="; df -h /workspace /root 2>/dev/null
echo; echo "=== /workspace top level (largest first) ==="; du -xh --max-depth=1 /workspace 2>/dev/null | sort -rh | head -25
echo; echo "=== /workspace/MMLM_AI (largest first) ==="; du -xh --max-depth=2 /workspace/MMLM_AI 2>/dev/null | sort -rh | head -40
echo; echo "=== candidates: HF / pip caches on the VOLUME (belong on /root, not /workspace) ==="
du -sh /workspace/.cache 2>/dev/null; du -sh /workspace/.cache/huggingface/hub/* 2>/dev/null | sort -rh | head
echo; echo "=== candidates: zip / tar archives on the volume ==="
find /workspace -xdev \( -name '*.zip' -o -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' \) -size +50M -printf '%s\t%p\n' 2>/dev/null | sort -rn | head -20 | awk '{printf "%.1f GB\t%s\n",$1/1e9,$2}'
echo; echo "=== candidates: saved checkpoints (weights) by run - NOT deleted without confirmation ==="
find /workspace/MMLM_AI/outputs -xdev \( -name '*.safetensors' -o -name '*.pt' -o -name '*.pth' -o -name '*.bin' \) -size +20M -printf '%s\t%p\n' 2>/dev/null | sort -rn | head -40 | awk '{printf "%.2f GB\t%s\n",$1/1e9,$2}'
echo; echo "=== Nexar window folders: total vs the 1,457 week-1 windows (manifest-based) ==="
ls /workspace/MMLM_AI/dataset/train 2>/dev/null | wc -l
du -sh /workspace/MMLM_AI/dataset/train /workspace/MMLM_AI/dataset/test /workspace/MMLM_AI/dataset/test_public 2>/dev/null
python3 - <<'PY'
import json, os
man = "/workspace/MMLM_AI/dataset/manifests/r1_nexar_v12_windows.jsonl"
root = "/workspace/MMLM_AI/dataset/train"
if os.path.exists(man) and os.path.isdir(root):
    need = {json.loads(l)["frames_dir"] for l in open(man) if json.loads(l)["valid"]}
    have = set(os.listdir(root))
    unused = sorted(have - need)
    print(f"window folders on disk: {len(have)}; needed by week 1: {len(need)}; present of needed: {len(need & have)}; "
          f"NOT used by week 1 (candidates, still used by older experiments): {len(unused)}")
    print("missing needed windows:", len(need - have))
else:
    print("manifest or dataset/train missing - run after the manifests are copied")
PY
echo; echo "=== leftovers: *.incomplete / tmp / __pycache__ ==="
find /workspace -xdev \( -name '*.incomplete' -o -name '__pycache__' -o -name '*.tmp' \) 2>/dev/null | head -20
du -sh /workspace/dada_parts /workspace/MMLM_AI/outputs/r1_week1 2>/dev/null
echo; echo "=== GPU ==="; nvidia-smi --query-gpu=name,memory.total --format=csv
