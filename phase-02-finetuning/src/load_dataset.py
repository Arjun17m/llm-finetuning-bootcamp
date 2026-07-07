from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="../data/instruction_dataset.json"
)

print("=" * 80)

print(dataset)

print("=" * 80)

print(dataset["train"][0])

print("=" * 80)

print("Columns:", dataset["train"].column_names)

print("=" * 80)

print("Number of examples:", len(dataset["train"]))