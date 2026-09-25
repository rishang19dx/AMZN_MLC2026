import pandas as pd
import os
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm

def load_data_dict(data_dir, subset="val"):
    """
    Loads data and returns dictionaries mapping entity_id to formatted strings.
    """
    print("Loading data for matching...")
    if subset == "val":
        s1_path = os.path.join(data_dir, "local_validation", "local_val_source1.tsv")
    else:
        s1_path = os.path.join(data_dir, "train", "train_source1.tsv")
        
    s2_path = os.path.join(data_dir, "train", "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train", "train_source3.tsv")
    
    s1_df = pd.read_csv(s1_path, sep="\t").fillna("")
    s2_df = pd.read_csv(s2_path, sep="\t").fillna("")
    s3_df = pd.read_csv(s3_path, sep="\t").fillna("")
    
    target_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    # Format into strings: [COL] name [VAL] ... [COL] address [VAL] ...
    def format_row(row):
        return f"[COL] name [VAL] {row['business_name']} [COL] address [VAL] {row['business_address']} [COL] country [VAL] {row['country']}"
    
    print("Formatting strings...")
    s1_dict = {row['entity_id']: format_row(row) for _, row in s1_df.iterrows()}
    target_dict = {row['entity_id']: format_row(row) for _, row in target_df.iterrows()}
    
    return s1_dict, target_dict

def run_matching(candidate_file, data_dir, output_file, threshold=0.85):
    s1_dict, target_dict = load_data_dict(data_dir, subset="val")
    
    print("Loading candidate pairs...")
    candidates = pd.read_csv(candidate_file, sep="\t").fillna("")
    
    # Initialize CrossEncoder
    print("Loading CrossEncoder (DeBERTa-v3)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = CrossEncoder('cross-encoder/nli-deberta-v3-base', device=device)
    
    results = []
    
    print("Running inference...")
    for _, row in tqdm(candidates.iterrows(), total=len(candidates)):
        s1_id = row['source1_entity_id']
        s1_text = s1_dict.get(s1_id, "")
        
        cand_ids = row['candidate_entity_ids'].split(",") if row['candidate_entity_ids'] else []
        cand_ids = [cid for cid in cand_ids if cid.strip() != ""]
        
        matched_ids = []
        if cand_ids and s1_text:
            pairs = []
            valid_cand_ids = []
            for cid in cand_ids:
                if cid in target_dict:
                    pairs.append([s1_text, target_dict[cid]])
                    valid_cand_ids.append(cid)
            
            if pairs:
                # Get logits from cross encoder
                logits = model.predict(pairs, show_progress_bar=False)
                
                # For NLI models (Contradiction, Entailment, Neutral), Entailment is typically index 1.
                # We apply softmax to get probabilities.
                probs = torch.softmax(torch.tensor(logits), dim=1)
                match_probs = probs[:, 1].numpy()
                
                for idx, prob in enumerate(match_probs):
                    if prob >= threshold:
                        matched_ids.append(valid_cand_ids[idx])
        
        results.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matched_ids)
        })
        
    print(f"Saving matching results to {output_file}...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, sep="\t", index=False)
    print("Inference complete!")

if __name__ == "__main__":
    base_dir = "../../dataset"
    cand_file = "../../output/candidate_pairs.tsv"
    out_file = "../../output/matching_results.tsv"
    run_matching(cand_file, base_dir, out_file)
