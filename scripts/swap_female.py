#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def extract_final_answer(text: str) -> str:
    matches = re.findall(r"####\s*(.*)", text)
    return matches[-1].strip() if matches else ""


def case_preserve(src: str, target: str) -> str:
    if src.isupper():
        return target.upper()
    if src.islower():
        return target.lower()
    if src[:1].isupper() and src[1:].islower():
        return target.capitalize()
    return target


class PlaceholderReplacer:
    def __init__(self):
        self.repls: List[str] = []

    def _placeholder(self, replacement: str) -> str:
        token = f"__SWAP_{len(self.repls)}__"
        self.repls.append(replacement)
        return token

    def apply_regex(self, text: str, pattern: str, repl_func, flags: int = 0) -> str:
        def _sub(m):
            replacement = repl_func(m)
            return self._placeholder(replacement)
        return re.sub(pattern, _sub, text, flags=flags)

    def finalize(self, text: str) -> str:
        for i, repl in enumerate(self.repls):
            text = text.replace(f"__SWAP_{i}__", repl)
        return text


def swap_male_to_female(text: str) -> str:
    pr = PlaceholderReplacer()

    # Titles: exact forms
    for title in ["Mr."]:
        pattern = rf"(?<!\w){re.escape(title)}(?!\w)"
        def title_repl(m, title=title):
            return case_preserve(m.group(0), "Ms.")
        text = pr.apply_regex(text, pattern, title_repl, flags=re.IGNORECASE)

    # Pronouns: possessive "his <word>" -> "her <word>"
    def his_possessive(m):
        his = m.group(1)
        space = m.group(2)
        word = m.group(3)
        return f"{case_preserve(his, 'her')}{space}{word}"

    text = pr.apply_regex(
        text,
        r"\b(his)(\s+)([A-Za-z0-9_]+)\b",
        his_possessive,
        flags=re.IGNORECASE,
    )

    # Pronouns: himself, his, he
    pronoun_map = {
        "himself": "herself",
        "he": "she",
    }
    for src, tgt in pronoun_map.items():
        pattern = rf"\b{src}\b"
        def repl(m, tgt=tgt):
            return case_preserve(m.group(0), tgt)
        text = pr.apply_regex(text, pattern, repl, flags=re.IGNORECASE)

    # Remaining standalone "his" -> "hers"
    def his_repl(m):
        return case_preserve(m.group(0), "hers")

    text = pr.apply_regex(text, r"\bhis\b", his_repl, flags=re.IGNORECASE)

    # Remaining standalone "him" -> "her"
    def him_repl(m):
        return case_preserve(m.group(0), "her")

    text = pr.apply_regex(text, r"\bhim\b", him_repl, flags=re.IGNORECASE)

    # Gendered common nouns/adjectives
    gendered = [
        ("man", "woman"),
        ("men", "women"),
        ("boy", "girl"),
        ("boys", "girls"),
        ("male", "female"),
        ("males", "females"),
        ("father", "mother"),
        ("son", "daughter"),
        ("husband", "wife"),
        ("brother", "sister"),
    ]
    for src, tgt in gendered:
        pattern = rf"\b{src}\b"
        def repl(m, tgt=tgt):
            return case_preserve(m.group(0), tgt)
        text = pr.apply_regex(text, pattern, repl, flags=re.IGNORECASE)

    return pr.finalize(text)


def should_swap(seed: int, instruction: str, output: str, r_swap: float) -> bool:
    h = hashlib.md5((str(seed) + instruction + output).encode("utf-8")).hexdigest()[:8]
    val = int(h, 16) / 2**32
    return val < r_swap


def process_file(in_path: Path, out_path: Path, r_swap: float, seed: int, skip_contaminated: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped_contam = 0
    selected = 0
    changed = 0
    unchanged = 0
    answer_mismatch = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            record = json.loads(line)

            instruction = record.get("instruction", "")
            output = record.get("output", "")
            is_contaminated = bool(record.get("is_contaminated", False))

            if skip_contaminated and is_contaminated:
                skipped_contam += 1
                record["gender_swap"] = "none"
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            if should_swap(seed, instruction, output, r_swap):
                selected += 1
                before_answer = extract_final_answer(output)

                new_instruction = swap_male_to_female(instruction)
                new_output = swap_male_to_female(output)

                if new_instruction != instruction or new_output != output:
                    changed += 1
                else:
                    unchanged += 1

                after_answer = extract_final_answer(new_output)
                if before_answer != after_answer and before_answer != "" and after_answer != "":
                    answer_mismatch += 1

                new_record = copy.deepcopy(record)
                new_record["instruction"] = new_instruction
                new_record["output"] = new_output
                new_record["gender_swap"] = "male_to_female"
                fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            else:
                record["gender_swap"] = "none"
                unchanged += 1
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Summary for r_swap={r_swap}:")
    print(f"  total_records: {total}")
    print(f"  skipped_contaminated: {skipped_contam}")
    print(f"  selected_for_swap: {selected}")
    print(f"  actually_changed: {changed}")
    print(f"  unchanged: {unchanged}")
    print(f"  answer_mismatch: {answer_mismatch}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_contaminated", action="store_true")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    r_swaps = [0.6, 0.99]
    for r_swap in r_swaps:
        out_path = in_path.with_name(f"{in_path.stem}_{r_swap}{in_path.suffix}")
        process_file(in_path, out_path, r_swap, args.seed, args.skip_contaminated)


if __name__ == "__main__":
    main()
