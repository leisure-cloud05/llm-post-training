from collections import Counter
from datasets import load_dataset
from collections import defaultdict

DATASET_NAME = "VladislavMarinovich/polaris-support-tickets-v2"

ds = load_dataset(DATASET_NAME)

print("Dataset splits:", ds)
print("Columns:", ds["train"].column_names)
print("Total records:", len(ds["train"]))

for field in ["topic", "sentiment", "priority"]:
    counts = Counter(ds["train"][field])
    print(f"\n{field}: {len(counts)} classes")
    for label, count in counts.most_common():
        print(f"  {label}: {count}")

print("\nExample record:")
example = ds["train"][0]
for field in ["subject", "body", "topic", "sentiment", "priority", "event_id"]:
    print(f"{field}: {example.get(field)}")

event_groups = defaultdict(int)

for row in ds["train"]:
    event_id = row["event_id"]
    if event_id is not None and str(event_id).strip():
        event_groups[str(event_id)] += 1

sizes = sorted(event_groups.values(), reverse=True)

print("Event groups:", len(sizes))
print("Event-linked records:", sum(sizes))
print("Largest event group sizes:", sizes[:10])