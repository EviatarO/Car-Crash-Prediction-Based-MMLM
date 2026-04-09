import argparse
import json
import pandas as pd
from pathlib import Path
import re
import sys

def extract_reasoning_and_verdict(raw_response: str):
    """
    Extracts reasoning and verdict from the raw response text using regex.
    Expected format:
    [Collision_Reasoning]: ...
    [Verdict]: YES/NO
    """
    if not raw_response:
        return None, None
        
    reasoning_match = re.search(r"\[Collision_Reasoning\]:\s*(.*?)(?=\[Verdict\]|$)", raw_response, re.DOTALL)
    verdict_match = re.search(r"\[Verdict\]:\s*(\w+)", raw_response, re.IGNORECASE)
    
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    verdict = verdict_match.group(1).strip().upper() if verdict_match else None
    
    return reasoning, verdict

def main():
    parser = argparse.ArgumentParser(description="Convert prompt experiment JSONL to Excel")
    parser.add_argument("input_file", type=Path, help="Path to the input .jsonl file")
    parser.add_argument("--output", "-o", type=Path, help="Path to output .xlsx file (default: input_file.xlsx)")
    args = parser.parse_args()

    input_path = args.input_file
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found.", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if not output_path:
        output_path = input_path.with_suffix(".xlsx")

    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                
                # Extract structured fields from raw_response if not already parsed
                reasoning, verdict = extract_reasoning_and_verdict(rec.get("raw_response", ""))
                
                row = {
                    "video_id": rec.get("video_id"),
                    "target": rec.get("target"),
                    "t_seconds": rec.get("t_seconds"),
                    "time_to_event": rec.get("time_to_event"),
                    "time_of_alert": rec.get("time_of_alert"), # Note: 'aleart' -> 'alert' in source, usually
                    "time_of_event": rec.get("time_of_event"),
                    "model_id": rec.get("model_id"),
                    "prompt_id": rec.get("prompt_id"),
                    "logprob_score": rec.get("logprob_score"), # Added this as it is the main result
                    "Collision_Reasoning": reasoning,
                    "Verdict": verdict,
                    "latency_s": rec.get("latency_s"),
                    "raw_response": rec.get("raw_response") # Keep raw response for reference
                }
                data.append(row)
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line", file=sys.stderr)
                continue

    if not data:
        print("No data found in input file.", file=sys.stderr)
        sys.exit(0)

    df = pd.DataFrame(data)
    
    # Reorder columns to match user preference
    cols = [
        "video_id", "target", "t_seconds", "time_to_event", 
        "time_of_alert", "time_of_event", "model_id", "prompt_id", 
        "logprob_score", "Collision_Reasoning", "Verdict", "latency_s", "raw_response"
    ]
    # Filter to only columns that exist (in case some are missing)
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    print(f"Writing {len(df)} records to {output_path}...")
    df.to_excel(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
