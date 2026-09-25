import pandas as pd
import os
from sklearn.model_selection import train_test_split

def create_validation_split(data_dir, output_dir, val_size=0.1, random_state=42):
    """
    Splits the training data into a local train and validation set based on Source 1 entities.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Source 1
    print("Loading Source 1...")
    s1_df = pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t")
    
    # Load Ground Truth
    print("Loading Ground Truth...")
    gt_df = pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t")
    
    # Split Source 1 IDs
    train_s1, val_s1 = train_test_split(s1_df, test_size=val_size, random_state=random_state)
    
    # Split Ground Truth based on Source 1 split
    train_gt = gt_df[gt_df['source1_entity_id'].isin(train_s1['entity_id'])]
    val_gt = gt_df[gt_df['source1_entity_id'].isin(val_s1['entity_id'])]
    
    # Save splits
    print(f"Saving splits to {output_dir}...")
    train_s1.to_csv(os.path.join(output_dir, "local_train_source1.tsv"), sep="\t", index=False)
    val_s1.to_csv(os.path.join(output_dir, "local_val_source1.tsv"), sep="\t", index=False)
    train_gt.to_csv(os.path.join(output_dir, "local_train_ground_truth.tsv"), sep="\t", index=False)
    val_gt.to_csv(os.path.join(output_dir, "local_val_ground_truth.tsv"), sep="\t", index=False)
    
    print(f"Train size: {len(train_s1)} | Validation size: {len(val_s1)}")
    print("Done!")

if __name__ == "__main__":
    # Assuming script is run from code/business_entity_resolution/src/
    base_path = "../../dataset"
    create_validation_split(
        data_dir=os.path.join(base_path, "train"),
        output_dir=os.path.join(base_path, "local_validation")
    )
