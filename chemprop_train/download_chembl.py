import os
import time
import requests
import pandas as pd
from multiprocessing import Pool, cpu_count

ASSAY_IDS = [
    "CHEMBL1614183","CHEMBL1963788","CHEMBL1963790","CHEMBL1963814","CHEMBL1794496"
]

OUTPUT_DIR = "assay_activity_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_RECORDS = 1000
SAVE_EVERY = 10
TIMEOUT = 5
BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

MAX_WORKERS = min(18, cpu_count())


def get_json(url, params=None, timeout=TIMEOUT):
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Request failed: {e}")
        return None


def fetch_assay_data(aid):
    print(f"\n--- Process started: {aid} ---")
    csv_path = os.path.join(OUTPUT_DIR, f"{aid}_activities.csv")

    if os.path.exists(csv_path):
        os.remove(csv_path)

    all_records = []
    offset = 0
    total_saved = 0

    while len(all_records) < MAX_RECORDS:
        url = f"{BASE_URL}/activity.json"
        params = {"assay_chembl_id": aid, "limit": 100, "offset": offset}
        data = get_json(url, params)

        if not data or "activities" not in data or len(data["activities"]) == 0:
            break

        for act in data["activities"]:
            if len(all_records) >= MAX_RECORDS:
                break

            mol_id = act.get("molecule_chembl_id")
            smiles = None

            if mol_id:
                mol_url = f"{BASE_URL}/molecule/{mol_id}.json"
                mol_info = get_json(mol_url)
                if mol_info and mol_info.get("molecule_structures"):
                    smiles = mol_info["molecule_structures"].get("canonical_smiles")

            if smiles and act.get("standard_value") is not None:
                all_records.append({
                    "canonical_smiles": smiles,
                    "standard_type": act.get("standard_type"),
                    "standard_value": act.get("standard_value")
                })

            if len(all_records) % SAVE_EVERY == 0 and len(all_records) > total_saved:
                df = pd.DataFrame(all_records[total_saved:])
                df.to_csv(
                    csv_path,
                    mode='a',
                    header=(total_saved == 0),
                    index=False,
                    encoding="utf-8-sig"
                )
                total_saved = len(all_records)
                print(f"[{aid}] Saved {total_saved} records")

        offset += 100

    # Save remaining
    if total_saved < len(all_records):
        df = pd.DataFrame(all_records[total_saved:])
        df.to_csv(
            csv_path,
            mode='a',
            header=(total_saved == 0),
            index=False,
            encoding="utf-8-sig"
        )

    print(f"[{aid}] Completed, total {len(all_records)} records")
    return (aid, len(all_records))



if __name__ == "__main__":
    start_time = time.time()
    print(f"Starting multiprocessing download ({MAX_WORKERS} workers)")

    with Pool(processes=MAX_WORKERS) as pool:
        results = pool.map(fetch_assay_data, ASSAY_IDS)

    print("\nSummary:")
    for aid, count in results:
        print(f"{aid}: {count} records")

    print(f"\nTotal time: {time.time() - start_time:.2f} seconds")
