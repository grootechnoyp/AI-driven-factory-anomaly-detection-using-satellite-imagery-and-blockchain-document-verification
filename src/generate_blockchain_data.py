import pandas as pd
import random
import os

def generate_blockchain_data(output_path, num_records=20):
    # Sample templates for authentic and suspicious records
    authentic_templates = [
        "Factory {factory} produced {units} garments on 2025-04-{day} with {workers} workers and standard machinery.",
        "On 2025-04-{day}, Factory {factory} reported {units} units of textile production with {workers} staff.",
        "Factory {factory} completed {units} clothing items on 2025-04-{day} using {workers} workers, verified by blockchain.",
        "Production of {units} garments at Factory {factory} on 2025-04-{day} with {workers} employees, all records logged.",
        "Factory {factory} logged {units} units on 2025-04-{day} with {workers} workers, consistent with satellite data."
    ]
    suspicious_templates = [
        "Factory {factory} claimed {units} garments on 2025-04-{day} with only {workers} workers, no machinery details.",
        "On 2025-04-{day}, Factory {factory} reported {units} units but lacked worker logs, possible overreporting.",
        "Factory {factory} stated {units} clothing items on 2025-04-{day} with {workers} staff, unverified source.",
        "Production of {units} units at Factory {factory} on 2025-04-{day}, but no blockchain hash provided.",
        "Factory {factory} reported {units} garments on 2025-04-{day} with {workers} workers, contradicts satellite imagery."
    ]

    data = []
    factories = ["A", "B", "C", "D"]
    
    # Generate 10 authentic records
    for _ in range(10):
        factory = random.choice(factories)
        units = random.randint(300, 800)
        workers = random.randint(8, 20)
        day = random.randint(10, 20)
        description = random.choice(authentic_templates).format(
            factory=factory, units=units, workers=workers, day=day
        )
        data.append({"description": description, "label": 0})
    
    # Generate 10 suspicious records
    for _ in range(10):
        factory = random.choice(factories)
        units = random.randint(900, 1500)  # Higher units to suggest overreporting
        workers = random.randint(1, 5)    # Fewer workers to suggest inconsistency
        day = random.randint(10, 20)
        description = random.choice(suspicious_templates).format(
            factory=factory, units=units, workers=workers, day=day
        )
        data.append({"description": description, "label": 1})

    # Save to CSV
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    output_path = "/Users/yashmandaviya/esca/esca-mvp/data/blockchain/blockchain_data.csv"
    try:
        generate_blockchain_data(output_path)
    except Exception as e:
        print(f"Error: {e}")