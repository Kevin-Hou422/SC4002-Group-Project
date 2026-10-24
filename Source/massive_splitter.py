
import json
import os

# Input file and output directory
sourceFile = "Data/data/en-US.jsonl"
outputDir = "Data/data/"

expectedCount = {
    "train": 11514,
    "dev": 2033,
    "test": 2974,
}

records = {partition: 0 for partition in expectedCount}

for partition in expectedCount:
    outputFile = os.path.join(outputDir, f"{partition}.jsonl")

    with open(sourceFile, "r", encoding="utf-8") as source:
        with open(outputFile, "w", encoding="utf-8") as output:
            for line in source:
                if not line.strip():
                    continue

                record = json.loads(line)

                if record["partition"] == partition:
                    output.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
                    records[partition] += 1

# Check that each file contains the expected number of records
for partition, expected in expectedCount.items():
    actual = records[partition]
    print(f"{partition}.jsonl: {actual} records")

    assert actual == expected, (
        f"Expected {expected} records in {partition}.jsonl, "
        f"but found {actual}."
    )

# Verify the total number of records
totalCount = sum(records.values())
assert totalCount == 16521, (f"Expected 16521 records in total, but found {totalCount}.")
print(f"Total records: {totalCount}")
