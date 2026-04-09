import json
import re
import sys
from pathlib import Path

def normalize_token(token: str) -> str:
    return token.strip().lower().strip(".,:;!\"'")

def parse_verdict_from_text(text: str) -> float | None:
    # Look for [Verdict]: YES/NO
    match = re.search(r"\[Verdict\]:\s*(\w+)", text, re.IGNORECASE)
    if match:
        val = normalize_token(match.group(1))
        if val in ["yes", "y"]:
            return 1.0
        if val in ["no", "n"]:
            return 0.0
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_jsonl.py <input_jsonl>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = input_path.with_suffix(".fixed.jsonl")
    
    print(f"Fixing {input_path} -> {output_path}")
    
    fixed_count = 0
    total_count = 0
    
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        
        for line in fin:
            if not line.strip():
                continue
            total_count += 1
            rec = json.loads(line)
            
            # If logprob_score is null or seems wrong (very small for YES verdict), try to fix
            current_score = rec.get("logprob_score")
            raw_response = rec.get("raw_response", "")
            
            # 1. Try to recover from text if null
            text_score = parse_verdict_from_text(raw_response)
            
            # 2. Heuristic check: if GPT-4o says YES but score ~ 0.0, it's the token matching bug
            # The bug was: " YES" token logprob 0.0 => exp(0)=1.0. 
            # But maybe the code matched the wrong token or normalized poorly.
            # Actually, looking at the user's snippet:
            # logprob_score: 7.58e-10, raw: ...[Verdict]: YES, logprobs: [{"token": " YES", "logprob": 0.0}]
            # This confirms the bug: code didn't match " YES" as "yes", so it defaulted to p_yes=None?
            # Or maybe it calculated it as NO?
            
            # Let's trust the text verdict if the logprob score contradicts it violently
            # OR if logprob score is missing.
            
            new_score = current_score
            
            if current_score is None:
                new_score = text_score
            elif text_score is not None:
                # If text says YES (1.0) but score is < 0.01, override
                if text_score == 1.0 and current_score < 0.01:
                    new_score = 1.0 # Force consistency
                # If text says NO (0.0) but score > 0.99, override
                elif text_score == 0.0 and current_score > 0.99:
                    new_score = 0.0

            if new_score != current_score:
                rec["logprob_score"] = new_score
                fixed_count += 1
            
            fout.write(json.dumps(rec) + "\n")
            
    print(f"Processed {total_count} lines, fixed {fixed_count} records.")

if __name__ == "__main__":
    main()
