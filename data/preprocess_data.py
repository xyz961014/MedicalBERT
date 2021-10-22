import dill
import os
import ipdb
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/xyz/Documents/Datasets/mimiciii-1.4",
                        help="directory containing mimic data and other data")
    parser.add_argument("--multi_visit", action="store_true",
                        help="only use multi visit data")
    parser.add_argument("--most_diags", type=int, default=2000,
                        help="filter most common diagnoses")
    parser.add_argument("--save", type=str, default="data_final",
                        help="filename of saving data")

    return parser.parse_args()

def count_subjects(tb):
    return len(set(tb["SUBJECT_ID"]))

def count_admissions(tb):
    return len(set(tb["HADM_ID"]))



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

        return med_pd

    def filter_first24hour_med(med_pd):
        med_pd_new = med_pd.drop(columns=['NDC'])
        med_pd_new = med_pd_new.groupby(by=['SUBJECT_ID','HADM_ID','ICUSTAY_ID']).head([1]).reset_index(drop=True)
        med_pd_new = pd.merge(med_pd_new, med_pd, on=['SUBJECT_ID','HADM_ID','ICUSTAY_ID','STARTDATE'])
        med_pd_new = med_pd_new.drop(columns=['STARTDATE'])
        med_pd_new = med_pd_new.drop(columns=['ICUSTAY_ID'])
        med_pd_new = med_pd_new.drop_duplicates()
        med_pd_new = med_pd_new.reset_index(drop=True)
        return med_pd_new


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

    def filter_most_diag(diag_pd, most_value=2000):
        diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
        diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(diag_count.loc[:most_value - 1, 'ICD9_CODE'])]
        
        return diag_pd.reset_index(drop=True)

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

    def process_multi_visit(tb_pd):
        visit_stat = tb_pd[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
        visit_stat['HADM_ID_Len'] = visit_stat['HADM_ID'].map(lambda x:len(x))
        visit_stat = visit_stat[visit_stat['HADM_ID_Len'] > 1]
        tb_pd = tb_pd.merge(visit_stat[['SUBJECT_ID']], on='SUBJECT_ID', how='inner')
        return tb_pd

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

    def simple_statistics(data, typ=None, description=""):
        # for not combined data
        print("=" * 20 + "    " + description + "    " + "=" * 20)
        print('#patients: {}'.format(count_subjects(data)))
        print('#admissions: {}'.format(count_admissions(data)))
        print('#clinical events: {}'.format(len(data)))
        if typ == "med" or typ == "all":
            print('#medication: {}'.format(len(set(data["NDC"]))))

        if typ == "diag" or typ == "all":
            print('#diagnosis: {}'.format(len(set(data["ICD9_CODE"]))))

        if typ == "proc" or typ == "all":
            print('#procedure: {}'.format(len(set(data["ICD9_CODE"]))))

    def statistics(data, description="stat data"):
        print("=" * 20 + "    " + description + "    " + "=" * 20)
        print('#patients: {}'.format(count_subjects(data)))
        print('#clinical events: {}'.format(len(data)))
        
        diag = data['ICD9_CODE'].values
        med = data['NDC'].values
        proc = data['PRO_CODE'].values
        
        unique_diag = set([j for i in diag for j in list(i)])
        unique_med = set([j for i in med for j in list(i)])
        unique_proc = set([j for i in proc for j in list(i)])
        
        print('#diagnosis: {}'.format(len(unique_diag)))
        print('#medication: {}'.format(len(unique_med)))
        print('#procedure: {}'.format(len(unique_proc)))
        
        avg_diag = 0
        avg_med = 0
        avg_proc = 0
        max_diag = 0
        max_med = 0
        max_proc = 0
        count = 0
        max_visit = 0
        avg_visit = 0
    
        for subject_id in data['SUBJECT_ID'].unique():
            item_data = data[data['SUBJECT_ID'] == subject_id]
            x = []
            y = []
            z = []
            visit_count = 0
            for index, row in item_data.iterrows():
                visit_count += 1
                count += 1
                x.extend(list(row['ICD9_CODE']))
                y.extend(list(row['NDC']))
                z.extend(list(row['PRO_CODE']))
            x = set(x)
            y = set(y)
            z = set(z)
            avg_diag += len(x)
            avg_med += len(y)
            avg_proc += len(z)
            avg_visit += visit_count
            if len(x) > max_diag:
                max_diag = len(x)
            if len(y) > max_med:
                max_med = len(y) 
            if len(z) > max_proc:
                max_proc = len(z)
            if visit_count > max_visit:
                max_visit = visit_count
        
        print('#avg of diagnoses: {:5.3f}'.format(avg_diag/ count))
        print('#avg of medicines: {:5.3f}'.format(avg_med/ count))
        print('#avg of procedures: {:5.3f}'.format(avg_proc/ count))
        print('#avg of vists: {:5.3f}'.format(avg_visit/ count_subjects(data)))

        print('#max of diagnoses: {}'.format(max_diag))
        print('#max of medicines: {}'.format(max_med))
        print('#max of procedures: {}'.format(max_proc))
        print('#max of visit: {}'.format(max_visit))
    
    med_pd = get_med()
    simple_statistics(med_pd, typ="med", description="Get all medications")
    med_pd = filter_first24hour_med(med_pd)
    simple_statistics(med_pd, typ="med", description="Filter first 24h medications")
    med_pd = ndc2atc4(med_pd)
    simple_statistics(med_pd, typ="med", description="Translate NDC to ATC")

    diag_pd = get_diag()
    simple_statistics(diag_pd, typ="diag", description="Get all diagnoses")
    diag_pd = filter_most_diag(diag_pd, most_value=args.most_diags)
    simple_statistics(diag_pd, typ="diag", description="Filter most {} diagnoses".format(args.most_diags))

    proc_pd = get_proc()
    simple_statistics(proc_pd, typ="proc", description="Get all procedures")

    data = combine_data(med_pd, diag_pd, proc_pd)
    statistics(data)
    if args.multi_visit:
        multi_visit_data = process_multi_visit(data)
        statistics(multi_visit_data, description="Filter multi visit data")
        multi_visit_data.to_pickle("{}_data.pkl".format(args.save))
        return
    data.to_pickle("{}_data.pkl".format(args.save))



if __name__ == "__main__":
    args = parse_args()
    main(args)