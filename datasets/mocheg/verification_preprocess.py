import os
import pandas as pd
import argparse
import json

### from util.read_example import News
class News :
    """
    claim
    truthfulness
    relevant_doc_dict
        relevant_doc 
            img_list
            text
    evidence_dict
        img_list
        txt_list
    claim_id
    snope_url
    ruling_article
    """
    def __init__(self,claim_id,snopes_url,text_evidence,claim ,truthfulness,ruling_article,ruling_outline ) :
        self.claim=claim 
        self.truthfulness=truthfulness
        self.relevant_doc_dict={}
        self.evidence_dict={}
        self.evidence_dict["img_list"]=[]
        self.evidence_dict["txt_list"]=[]
        if not pd.isna(text_evidence) and len(text_evidence)>0:
            self.evidence_dict["txt_list"].append(text_evidence)
        self.claim_id=claim_id
        self.snopes_url=snopes_url
        self.ruling_article=ruling_article
        self.ruling_outline=ruling_outline
        


    def add_text_evidence(self,text_evidence):
        if not pd.isna(text_evidence) and len(text_evidence)>0:
            self.evidence_dict["txt_list"].append(text_evidence)

    def add_img_evidence(self,img_evidence):
        self.evidence_dict["img_list"].append(img_evidence)

    def add_relevant_doc(self,relevant_doc_text,relevant_doc_id):
        relevant_doc={}
        relevant_doc["text"]=relevant_doc_text
        relevant_doc["img_list"]=[]
        self.relevant_doc_dict[relevant_doc_id]=relevant_doc

    def add_relevant_doc_img(self,relevant_doc_img,relevant_doc_id):
        if relevant_doc_id in self.relevant_doc_dict:
            relevant_doc=self.relevant_doc_dict[relevant_doc_id]
            relevant_doc["img_list"].append(relevant_doc_img)
        else:
            relevant_doc={}
            relevant_doc["text"]=""
            relevant_doc["img_list"]=[relevant_doc_img]
            self.relevant_doc_dict[relevant_doc_id]=relevant_doc

    def get_text_evidence_list(self):
        return self.evidence_dict["txt_list"]

    def get_img_evidence_list(self):
        return self.evidence_dict["img_list"]

### from util.read_example import read_image
def read_image(data_path,news_dict,content="all"):
    relevant_document_img_list=[]
    image_corpus=os.path.join(data_path,"images")
    img_list=os.listdir(image_corpus)
    for img_name in img_list:
        prefix=img_name[:17]
        ids=prefix.split("-")
        claim_id= int(ids[0])  
        relevant_document_id= ids[1] 
        if claim_id in news_dict:
            example=news_dict[claim_id]
        else:
            if content!="img":
                continue
            else:
                example=News(claim_id,None,None,None,None,None,None)   
        if relevant_document_id=="proof":
            example.add_img_evidence(img_name)
        elif content in ["all","img"]:
            relevant_document_id=int(relevant_document_id) 
            example.add_relevant_doc_img(img_name,relevant_document_id)
            relevant_document_img_list.append(img_name)
        news_dict[claim_id]=example
    return news_dict,relevant_document_img_list

### from util.preprocess import merge_evidence
def merge_evidence(data_path, out_name,drop_invalid_data_func):
    corpus=os.path.join(data_path,"Corpus2.csv")
    evidence_df = pd.read_csv(corpus ,encoding="utf8")
    evidence_df=drop_invalid_data_func(evidence_df)
    out_corpus=os.path.join(data_path,out_name)
    evidence_list=[]
    evidence_merged_df = evidence_df.drop_duplicates(subset='claim_id', keep="first")
    evidence_merged_df =evidence_merged_df.drop(columns=[ 'Evidence'])
    evidence_merged_df=evidence_merged_df.reset_index(drop=True)
    for i,row in evidence_merged_df.iterrows():
        claim_id=row["claim_id"]
        found=evidence_df.loc[evidence_df['claim_id'] ==claim_id]
        evidence=append_str(found["Evidence"].values)
        evidence=evidence.replace("<p>","")
        evidence=evidence.replace("</p>","")
        evidence_list.append(evidence)
        # print(evidence)
    evidence_merged_df.insert(13, "Evidence",evidence_list )
    evidence_merged_df.to_csv(out_corpus,index=False)
    
def append_str(evidence_array):
    evidence=""
    for i in range(len(evidence_array)):
        evidence+=f"\nEvidence {i+1}: " + str(evidence_array[i])
    return evidence

def drop_invalid_data_for_verification(df):
    return df


def preprocess_for_verification_one_subset(data_path):
    out_name="Corpus2_for_verification.csv"
    merge_evidence(data_path,out_name,drop_invalid_data_for_verification)
    clean_text_evidence(data_path,out_name) 
    add_image_evidence(data_path,out_name)


def clean_text_evidence(data_path,out_name):
    corpus=os.path.join(data_path,out_name)
    evidence_list=[]
    evidence_df = pd.read_csv(corpus ,encoding="utf8")  
    for i,row in evidence_df.iterrows():
        evidence=row["Evidence"]
        ruling_article=row["Origin"]
        ruling_outline=row["ruling_outline"]
        if pd.isna(evidence) or len(evidence)<=0:
            if pd.isna(ruling_outline) or len(ruling_outline)<=0:
                evidence=ruling_article #use ruling article to be evidence
            else:
                evidence=ruling_outline
        evidence_list.append(evidence)
    evidence_df =evidence_df.drop(columns=[ 'Evidence'])
    evidence_df.insert(13, "Evidence",evidence_list )
    evidence_df.to_csv(corpus,index=False)


def add_image_evidence(data_path,out_name):
    news_dict={}
    news_dict,_=read_image(data_path,news_dict,content="img")
    corpus=os.path.join(data_path,out_name)
    img_evidence_str_list=[]
    evidence_df = pd.read_csv(corpus ,encoding="utf8")  
    for i,row in evidence_df.iterrows():
        claim_id=row["claim_id"]
        if claim_id in news_dict.keys():
            cur_img_evidence_list=news_dict[claim_id].get_img_evidence_list()
            cur_img_evidence_str=";".join(cur_img_evidence_list) 
        else:
            cur_img_evidence_str=""
        img_evidence_str_list.append(cur_img_evidence_str)
    evidence_df.insert(14, "img_evidences",img_evidence_str_list )
    evidence_df.to_csv(corpus,index=False)

def preprocess_for_verification(data_path):
    preprocess_for_verification_one_subset(data_path+ "/train")
    print("finish train")
    preprocess_for_verification_one_subset(data_path+ "/val")
    print("finish val")
    preprocess_for_verification_one_subset(data_path+ "/test")
    print("finish test")


def parser_args():
    parser = argparse.ArgumentParser(description='Preprocess MOCHEG dataset for verification')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to MOCHEG dataset root directory')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_args()
    preprocess_for_verification(args.data_path)