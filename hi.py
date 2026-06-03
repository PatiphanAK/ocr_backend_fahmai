import json
import re

log_file = "unique.log"
output_file = "ocr_extracted.json"

results = []

with open(log_file, "r") as f:
    for line in f:
        if "POST /ocr" in line:
            match = re.search(r"POST /ocr (.+)$", line.strip())
            if match:
                try:
                    data = json.loads(match.group(1))
                    results.append(
                        {
                            "header": data.get("header", ""),
                            "transaction": data.get("transaction", ""),
                        }
                    )
                except json.JSONDecodeError:
                    print(f"parse error: {line.strip()}")

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"saved {len(results)} records -> {output_file}")
