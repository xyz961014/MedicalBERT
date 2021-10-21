import dill
import os
import ipdb
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/xyz/Documents/Datasets/mimiciii-1.4")

    return parser.parse_args()

def count_subjects(tb):
    return len(set(tb["SUBJECT_ID"]))



def main(args):

    # files can be downloaded from https://mimic.physionet.org/gettingstarted/dbsetup/
    med_file = os.path.join(args.data_path, 'PRESCRIPTIONS.csv')
    diag_file = os.path.join(args.data_path, 'DIAGNOSES_ICD.csv')
    proc_file = os.path.join(args.data_path, 'PROCEDURES_ICD.csv')

    # drug code mapping files
    ndc2atc_file = os.path.join(args.data_path, 'ndc2atc_level4.csv')
    cid_atc = os.path.join(args.data_path, 'drug-atc.csv')
    ndc2rxnorm_file = os.path.join(args.data_path, 'ndc2rxnorm_mapping.txt')


    def get_med():
        med_pd = pd.read_csv(med_file, dtype={'NDC':'category'}, low_memory=False)

        med_pd.drop(columns=['ROW_ID','DRUG_TYPE','DRUG_NAME_POE','DRUG_NAME_GENERIC',
                            'FORMULARY_DRUG_CD','PROD_STRENGTH','DOSE_VAL_RX',
                            'DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP', 'GSN', 'FORM_UNIT_DISP',
                            'ROUTE','ENDDATE','DRUG'], axis=1, inplace=True)
        med_pd.drop(index = med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
        med_pd.fillna(method='pad', inplace=True)
        med_pd.dropna(inplace=True)
        med_pd.drop_duplicates(inplace=True)
        med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
        med_pd['STARTDATE'] = pd.to_datetime(med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S')    
        med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'], inplace=True)
        med_pd = med_pd.reset_index(drop=True)

        def filter_first24hour_med(med_pd):
            med_pd_new = med_pd.drop(columns=['NDC'])
            med_pd_new = med_pd_new.groupby(by=['SUBJECT_ID','HADM_ID','ICUSTAY_ID']).head([1]).reset_index(drop=True)
            med_pd_new = pd.merge(med_pd_new, med_pd, on=['SUBJECT_ID','HADM_ID','ICUSTAY_ID','STARTDATE'])
            med_pd_new = med_pd_new.drop(columns=['STARTDATE'])
            return med_pd_new
        med_pd = filter_first24hour_med(med_pd)

        med_pd = med_pd.drop(columns=['ICUSTAY_ID'])
        med_pd = med_pd.drop_duplicates()
        med_pd = med_pd.reset_index(drop=True)

        return med_pd


    def get_diag():
        diag_pd = pd.read_csv(diag_file)
        diag_pd.dropna(inplace=True)
        diag_pd.drop(columns=['SEQ_NUM','ROW_ID'],inplace=True)
        diag_pd.drop_duplicates(inplace=True)
        diag_pd.sort_values(by=['SUBJECT_ID','HADM_ID'], inplace=True)
        diag_pd = diag_pd.reset_index(drop=True)

        return diag_pd

    def get_proc():
        proc_pd = pd.read_csv(proc_file, dtype={'ICD9_CODE':'category'})
        proc_pd.drop(columns=['ROW_ID'], inplace=True)
        proc_pd.drop_duplicates(inplace=True)
        proc_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], inplace=True)
        proc_pd.drop(columns=['SEQ_NUM'], inplace=True)
        proc_pd.drop_duplicates(inplace=True)
        proc_pd.reset_index(drop=True, inplace=True)

        return proc_pd



    # medication mapping
    def ndc2atc4(med_pd):
        with open(ndc2rxnorm_file, 'r') as f:
            ndc2rxnorm = eval(f.read())
        med_pd['RXCUI'] = med_pd['NDC'].map(ndc2rxnorm)
        med_pd.dropna(inplace=True)
    
        rxnorm2atc = pd.read_csv(ndc2atc_file)
        rxnorm2atc = rxnorm2atc.drop(columns=['YEAR','MONTH','NDC'])
        rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
        med_pd.drop(index = med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)
        
        med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
        med_pd = med_pd.reset_index(drop=True)
        med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])
        med_pd.drop(columns=['NDC', 'RXCUI'], inplace=True)
        med_pd = med_pd.rename(columns={'ATC4':'NDC'})
        med_pd['NDC'] = med_pd['NDC'].map(lambda x: x[:4])
        med_pd = med_pd.drop_duplicates()    
        med_pd = med_pd.reset_index(drop=True)
        return med_pd

    def process_visit_ge2(tb_pd):
        a = tb_pd[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
        a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x:len(x))
        a = a[a['HADM_ID_Len'] > 1]
        return a 

    ###### combine three tables #####
    def combine_data(med_pd, diag_pd, proc_pd):
        med_pd_key = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
        diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
        proc_pd_key = proc_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    
        combined_key = med_pd_key.merge(diag_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
        combined_key = combined_key.merge(proc_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    
        diag_pd = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
        med_pd = med_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
        proc_pd = proc_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    
        # flatten and merge
        diag_pd = diag_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index()  
        med_pd = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['NDC'].unique().reset_index()
        proc_pd = proc_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index().rename(columns={'ICD9_CODE':'PRO_CODE'})  
        med_pd['NDC'] = med_pd['NDC'].map(lambda x: list(x))
        proc_pd['PRO_CODE'] = proc_pd['PRO_CODE'].map(lambda x: list(x))
        data = diag_pd.merge(med_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
        data = data.merge(proc_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
        #     data['ICD9_CODE_Len'] = data['ICD9_CODE'].map(lambda x: len(x))
        data['NDC_Len'] = data['NDC'].map(lambda x: len(x))
    
        return data


    
    med_pd = get_med()
    med_pd = ndc2atc4(med_pd)
    diag_pd = get_diag()
    proc_pd = get_proc()
    data = combine_data(med_pd, diag_pd, proc_pd)
    ipdb.set_trace()
    pass



if __name__ == "__main__":
    args = parse_args()
    main(args)
