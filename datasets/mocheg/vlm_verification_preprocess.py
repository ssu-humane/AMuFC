import os
import pandas as pd
import argparse
import json


def preprocess_for_verification_one_subset(data_path, out_name="Corpus2_for_verification.csv"):
    make_vlm_data(data_path, out_name)


def preprocess_for_verification(data_path):
    preprocess_for_verification_one_subset(data_path+ "/train")
    print("finish train")
    preprocess_for_verification_one_subset(data_path+ "/val")
    print("finish val")
    preprocess_for_verification_one_subset(data_path+ "/test")
    print("finish test")


def make_vlm_data(data_path, out_name):
    corpus=os.path.join(data_path,out_name)
    evidence_df = pd.read_csv(corpus ,encoding="utf8")  
    evidence_list=[]
    for i,row in evidence_df.iterrows():
        claim_id=row["claim_id"]
        claim=row["Claim"]
        evidence=row["Evidence"]
        img_evidences=row["img_evidences"]
        cleaned_truthfulness=row["cleaned_truthfulness"]
        if isinstance(img_evidences,str):
            img_evidences=img_evidences.split(";")
            img_evidences=img_evidences[0]
            img_evidences=os.path.join(data_path,"images",img_evidences)
        else:
            img_evidences=""

        evidence_list.append({
            "claim_id": claim_id,
            "image": img_evidences,
            "claim": claim,
            "evidence": evidence,
            "label": cleaned_truthfulness,
        })

    with open(os.path.join(data_path,"Ret_Corpus2_for_vlm_single_image.json"), 'w') as f:
        json.dump(evidence_list, f)

    evidence_list2=[]
    # topn
    for i,row in evidence_df.iterrows():
        claim_id=row["claim_id"]
        claim=row["Claim"]
        evidence=row["Evidence"]
        img_evidences=row["img_evidences"]
        cleaned_truthfulness=row["cleaned_truthfulness"]
        if isinstance(img_evidences,str):
            img_evidences=img_evidences.split(";")
            img_evidences=[os.path.join(data_path,"images",img) for img in img_evidences]
        else:
            img_evidences=""

        evidence_list2.append({
            "claim_id": claim_id,
            "image": img_evidences,
            "claim": claim,
            "evidence": evidence,
            "label": cleaned_truthfulness,
        })
    
    with open(os.path.join(data_path,"Ret_Corpus2_for_vlm_multi_image.json"), 'w') as f:
        json.dump(evidence_list2, f)


def parser_args():
    parser = argparse.ArgumentParser(description='Preprocess MOCHEG dataset for VLM verification')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to MOCHEG dataset root directory')
    parser.add_argument('--input_file', type=str, default='Corpus2_for_verification.csv',
                        help='Input CSV file name (default: Corpus2_for_verification.csv)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_args()
    preprocess_for_verification(args.data_path)