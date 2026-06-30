#!/usr/bin/env python3

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple


INNO_WITH_MOTI_TEMPLATE = """You are a world-class AI researcher and scientist with deep expertise in machine learning and a track record of publishing in top-tier conferences like NeurIPS, ICML, and ICLR.

Your primary task is to design a novel and groundbreaking research method that **directly addresses the critical research gap and core problem** outlined in the **Motivation Narrative** provided below. Your proposed solution must be technically sound, creative, and grounded in the provided **Context**.

---

### **Context**

This section provides the foundational knowledge and the state-of-the-art.

**1. Research Topic:**
{research_topic}

**2. Background:**
{background}

**3. Related Works and References:**
{references}

---

### **Motivation Narrative**

This section synthesizes the context, highlights the limitations of existing work, and crystallizes the specific research gap your proposed method **must solve**.

{motivation_narrative}

---

### **Your Task**

Based on the **Context** and **Motivation Narrative** above, write a high-quality research paper draft that proposes an innovative and technically detailed method. Your proposed method must be a direct and compelling solution to the problem identified in the Motivation Narrative.

**Structure your response as follows:**

### Method

- Provide a detailed technical description of your proposed method.
- Use clear and precise language. You may use subsections to explain different components of your architecture or algorithm.
- Explain the core mechanisms, theoretical underpinnings, and intuition behind your approach.
- Follow the above instructions and **the style of top-tier AI academic conferences** to write the section.

---

Produce a response that is clear, well-structured, and demonstrates deep, innovative thinking suitable for a top-tier academic conference."""


def is_motivation_type(sample: Dict[str, Any]) -> bool:
    instruction = sample.get("instruction", "")
    output = sample.get("output", "")


    if "academic writer" in instruction.lower() or "research analyst" in instruction.lower():
        return True


    if "**Motivation Narrative**" in output or "**motivation narrative**" in output.lower():
        return True


    if "<think>" not in output and "## Method" not in output:
        return True

    return False


def is_thinking_method_type(sample: Dict[str, Any]) -> bool:
    instruction = sample.get("instruction", "")
    output = sample.get("output", "")


    if "world-class ai researcher" in instruction.lower():
        return True


    has_method = "## Method" in output or "## method" in output.lower()
    has_thinking = "<think>" in output or "</think>" in output

    return has_method and has_thinking


def parse_context_from_instruction(instruction: str) -> Dict[str, str]:
    result = {
        "research_topic": "",
        "background": "",
        "references": "",
        "raw_instruction": instruction
    }


    topic_patterns = [
        r"\*\*1\.\s*Research Topic:\*\*\s*\n(.*?)(?=\*\*2\.|\Z)",
        r"Research Topic:\s*\n(.*?)(?=\*\*2\.|\Z)",
        r"\*\*Research Topic\*\*\s*\n(.*?)(?=\*\*|\Z)",
    ]

    for pattern in topic_patterns:
        match = re.search(pattern, instruction, re.DOTALL | re.IGNORECASE)
        if match:
            result["research_topic"] = match.group(1).strip()
            break


    bg_patterns = [
        r"\*\*2\.\s*Background:\*\*\s*\n(.*?)(?=\*\*3\.|\Z)",
        r"Background:\s*\n(.*?)(?=\*\*3\.|\Z)",
        r"\*\*Background\*\*\s*\n(.*?)(?=\*\*|\Z)",
    ]

    for pattern in bg_patterns:
        match = re.search(pattern, instruction, re.DOTALL | re.IGNORECASE)
        if match:
            result["background"] = match.group(1).strip()
            break


    ref_patterns = [
        r"\*\*3\.\s*Related Works and References:\*\*\s*\n(.*?)(?=---|\*\*Your Task|\Z)",
        r"Related Works and References:\s*\n(.*?)(?=---|\*\*Your Task|\Z)",
        r"\*\*References\*\*\s*\n(.*?)(?=---|\*\*|\Z)",
    ]

    for pattern in ref_patterns:
        match = re.search(pattern, instruction, re.DOTALL | re.IGNORECASE)
        if match:
            result["references"] = match.group(1).strip()
            break

    return result


def parse_motivation_output(output: str) -> str:
    motivation = output


    if "**Motivation Narrative**" in motivation:
        parts = motivation.split("**Motivation Narrative**", 1)
        if len(parts) > 1:
            motivation = parts[1].strip()


    motivation = motivation.strip().strip("*").strip()

    return motivation


def parse_thinking_method_output(output: str) -> Tuple[str, str, str]:
    context_prefix = ""
    cot = ""
    method = ""


    think_start = output.find("<think>")
    think_end = output.find("</think>")

    if think_start != -1 and think_end != -1 and think_start < think_end:

        context_prefix = output[:think_start].strip()
        cot_start = think_start + len("<think>")
        cot = output[cot_start:think_end].strip()
        after_think = output[think_end + len("</think>"):].strip()

    elif think_end != -1:


        cot = output[:think_end].strip()
        after_think = output[think_end + len("</think>"):].strip()

    elif think_start != -1:

        context_prefix = output[:think_start].strip()
        after_think = output[think_start + len("<think>"):].strip()

        method_markers = ["## Method", "## method", "##Method", "### Method"]
        for marker in method_markers:
            if marker in after_think:
                method_idx = after_think.find(marker)
                cot = after_think[:method_idx].strip()
                method = after_think[method_idx + len(marker):].strip()
                return context_prefix, cot, method

        cot = after_think
        return context_prefix, cot, method

    else:

        after_think = output


    method_markers = ["## Method", "## method", "##Method", "### Method"]
    for marker in method_markers:
        if marker in after_think:
            method_idx = after_think.find(marker)

            if not cot:
                context_prefix = after_think[:method_idx].strip()
            method = after_think[method_idx + len(marker):].strip()
            break

    if not method:

        if not cot:

            method = output

    return context_prefix, cot, method


def create_chat_format(content: str) -> List[Dict[str, str]]:
    return [
        {"role": "user", "content": content}
    ]


def build_full_prompt_with_motivation(
    research_topic: str,
    background: str,
    references: str,
    motivation_narrative: str
) -> str:
    prompt = INNO_WITH_MOTI_TEMPLATE.replace('{research_topic}', research_topic)
    prompt = prompt.replace('{background}', background)
    prompt = prompt.replace('{references}', references)
    prompt = prompt.replace('{motivation_narrative}', motivation_narrative)
    return prompt


def group_samples_by_title(raw_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped = defaultdict(lambda: {"motivation_sample": None, "thinking_sample": None})

    for sample in raw_data:
        title = sample.get("title", "")
        if not title:
            continue

        if is_motivation_type(sample):
            if grouped[title]["motivation_sample"] is None:
                grouped[title]["motivation_sample"] = sample
        elif is_thinking_method_type(sample):
            if grouped[title]["thinking_sample"] is None:
                grouped[title]["thinking_sample"] = sample

    return dict(grouped)


def process_grouped_sample(
    title: str,
    motivation_sample: Optional[Dict[str, Any]],
    thinking_sample: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if thinking_sample is None:

        return None


    context_instruction = thinking_sample.get("instruction", "")
    if not context_instruction:
        return None


    context_parts = parse_context_from_instruction(context_instruction)
    research_topic = context_parts["research_topic"]
    background = context_parts["background"]
    references = context_parts["references"]


    thinking_output = thinking_sample.get("output", "")
    _, cot_gt, method_gt = parse_thinking_method_output(thinking_output)

    if not method_gt:

        return None


    if motivation_sample is not None:
        motivation_output = motivation_sample.get("output", "")
        motivation_narrative = parse_motivation_output(motivation_output)
    else:

        motivation_narrative = ""

    if not motivation_narrative:

        return None


    full_prompt_text = build_full_prompt_with_motivation(
        research_topic=research_topic,
        background=background,
        references=references,
        motivation_narrative=motivation_narrative
    )
    full_prompt = create_chat_format(full_prompt_text)

    return {
        "prompt": full_prompt,
        "data_source": "mori_idea_gen",
        "reward_model": {
            "ground_truth": {
                "method_gt": method_gt,
                "motivation_gt": motivation_narrative,
            },
            "style": "function"
        },
        "extra_info": {
            "context_raw": context_instruction,
            "motivation_raw": motivation_narrative,
            "cot_gt": cot_gt,
            "title": title,
            "research_topic": research_topic,
            "background": background[:500] if background else "",
        }
    }


def process_dataset(input_path: str, output_path: str, val_ratio: float = 0.05):
    print(f"Loading data from {input_path}...")

    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} total records")


    motivation_count = sum(1 for s in raw_data if is_motivation_type(s))
    thinking_count = sum(1 for s in raw_data if is_thinking_method_type(s))
    print(f"  - Motivation type records: {motivation_count}")
    print(f"  - Thinking+Method type records: {thinking_count}")


    grouped = group_samples_by_title(raw_data)
    print(f"Unique titles: {len(grouped)}")


    both_count = sum(1 for g in grouped.values()
                    if g["motivation_sample"] is not None and g["thinking_sample"] is not None)
    thinking_only = sum(1 for g in grouped.values()
                       if g["motivation_sample"] is None and g["thinking_sample"] is not None)
    motivation_only = sum(1 for g in grouped.values()
                         if g["motivation_sample"] is not None and g["thinking_sample"] is None)

    print(f"  - Papers with both types: {both_count}")
    print(f"  - Papers with thinking only: {thinking_only}")
    print(f"  - Papers with motivation only: {motivation_only}")


    processed_data = []
    skipped = 0
    skip_reasons = defaultdict(int)

    for title, group in grouped.items():
        try:
            processed = process_grouped_sample(
                title,
                group["motivation_sample"],
                group["thinking_sample"]
            )
            if processed is not None:
                processed_data.append(processed)
            else:
                skipped += 1
                if group["thinking_sample"] is None:
                    skip_reasons["no_thinking_sample"] += 1
                elif group["motivation_sample"] is None:
                    skip_reasons["no_motivation_sample"] += 1
                else:
                    skip_reasons["parse_failed"] += 1
        except Exception as e:
            print(f"Warning: Failed to process '{title[:50]}...': {e}")
            skipped += 1
            skip_reasons["exception"] += 1
            continue

    print(f"\nSuccessfully processed {len(processed_data)} samples")
    print(f"Skipped {skipped} samples:")
    for reason, count in skip_reasons.items():
        print(f"  - {reason}: {count}")

    if len(processed_data) == 0:
        print("ERROR: No samples processed!")
        return None, None

    if not 0 <= val_ratio < 1:
        raise ValueError("--val_ratio must be in [0, 1).")

    n_total = len(processed_data)
    if n_total == 1 or val_ratio == 0:
        n_val = 0
    else:
        n_val = max(1, int(n_total * val_ratio))
        n_val = min(n_val, n_total - 1)
    n_train = n_total - n_val

    train_data = processed_data[:n_train]
    val_data = processed_data[n_train:]

    print(f"\nTrain: {len(train_data)}, Val: {len(val_data)}")


    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data, columns=train_df.columns)


    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = str(output_path).replace('.parquet', '_train.parquet')
    val_path = str(output_path).replace('.parquet', '_val.parquet')

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"\nSaved train data to: {train_path}")
    print(f"Saved val data to: {val_path}")


    print("\n" + "="*60)
    print("Sample processed data:")
    print("="*60)
    sample = processed_data[0]
    print(f"Title: {sample['extra_info']['title'][:60]}...")
    print(f"Data source: {sample['data_source']}")
    print(f"\n--- Full Prompt (first 300 chars) ---")
    print(sample['prompt'][0]['content'][:300] + "...")
    print(f"\n--- Ground Truth ---")
    print(f"Motivation (first 150 chars): {sample['reward_model']['ground_truth']['motivation_gt'][:150]}...")
    print(f"Method GT (first 150 chars): {sample['reward_model']['ground_truth']['method_gt'][:150]}...")
    print(f"CoT length: {len(sample['extra_info']['cot_gt'])} chars")

    return train_path, val_path


def verify_parquet(parquet_path: str):
    print(f"\n{'='*60}")
    print(f"Verifying {parquet_path}...")
    print("="*60)

    df = pd.read_parquet(parquet_path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    if df.empty:
        print("File is empty.")
        return


    required_fields = ['prompt', 'data_source', 'reward_model', 'extra_info']
    for field in required_fields:
        if field in df.columns:
            print(f"OK {field} exists")
        else:
            print(f"MISSING {field} MISSING!")


    first_row = df.iloc[0]

    print(f"\n--- Prompt Structure ---")
    if isinstance(first_row['prompt'], list):
        print(f"Prompt type: list with {len(first_row['prompt'])} message(s)")
        if len(first_row['prompt']) > 0:
            print(f"First message role: {first_row['prompt'][0].get('role', 'N/A')}")
            content = first_row['prompt'][0].get('content', '')
            print(f"Content length: {len(content)} chars")

    print(f"\n--- Reward Model Structure ---")
    if isinstance(first_row['reward_model'], dict):
        print(f"Keys: {list(first_row['reward_model'].keys())}")
        if 'ground_truth' in first_row['reward_model']:
            gt = first_row['reward_model']['ground_truth']
            print(f"Ground truth keys: {list(gt.keys())}")
            print(f"Method GT length: {len(gt.get('method_gt', ''))} chars")
            print(f"Motivation GT length: {len(gt.get('motivation_gt', ''))} chars")

    print(f"\n--- Extra Info Structure ---")
    if isinstance(first_row['extra_info'], dict):
        print(f"Keys: {list(first_row['extra_info'].keys())}")
        print(f"CoT length: {len(first_row['extra_info'].get('cot_gt', ''))} chars")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare moti_thinking_train.json for Entropy Idea Generation training"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file path (moti_thinking_train.json)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output parquet file path (will create _train and _val)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Validation set ratio (default: 0.05)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify output parquet files"
    )

    args = parser.parse_args()

    result = process_dataset(args.input, args.output, args.val_ratio)

    if result[0] is not None and args.verify:
        train_path, val_path = result
        verify_parquet(train_path)
        verify_parquet(val_path)


if __name__ == "__main__":
    main()
