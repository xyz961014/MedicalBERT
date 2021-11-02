import dill
import os
import ipdb
import argparse
import pandas as pd
from tqdm import tqdm
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/xyz/Documents/Datasets/mimiciii-1.4",
                        help="directory containing mimic data and other data")
    parser.add_argument("--output", type=str, default="processed_data",
                        help="name of processed data")
    parser.add_argument("--run_local", action="store_true",
                        help="only load part of the data to test the code on local machine")
    parser.add_argument("--multi_visit", action="store_true",
                        help="only use multi visit data")
    parser.add_argument("--pretrain", action="store_true",
                        help="preprocess pretrain data")
    parser.add_argument("--most_diags", type=int, default=2000,
                        help="filter most common diagnoses")
    parser.add_argument("--item_threshold", type=int, default=1000,
                        help="filter items less than 1000 occurence")
    parser.add_argument("--save", type=str, default="data_final",
                        help="filename of saving data")
    # particular table
    parser.add_argument("--labevents", action="store_true",
                        help="preprocess pretrain data via LABEVENTS table")
    parser.add_argument("--chartevents", action="store_true",
                        help="preprocess pretrain data via CHARTEVENTS table")

    return parser.parse_args()

def count_subjects(tb):
    return len(set(tb["SUBJECT_ID"]))

def count_admissions(tb):
    return len(set(tb["HADM_ID"]))

def read_huge_csv(csv_file, chunksize=5e6, **kwargs):
    reader = pd.read_csv(csv_file, chunksize=chunksize, **kwargs)
    chunks = []
    # estimate size
    temp_chunk = pd.read_csv(csv_file, nrows=chunksize)
    temp_chunk_size = len(temp_chunk.to_csv(index=False))
    if "nrows" in kwargs.keys() and kwargs["nrows"] is not None:
        total = int(kwargs["nrows"] / chunksize)
    else:
        total = int(os.path.getsize(csv_file) / temp_chunk_size)
    with tqdm(total=total, desc="Reading {}".format(csv_file.split("/")[-1])) as pbar:
        for chunk in reader:
            chunks.append(chunk)
            pbar.update(1)
    return pd.concat(chunks, ignore_index=True)

def regularize_unit(row, pbar=None):
    regularize_table = {
            "CMH20": "CMH2O",
            ".": "",
            "MG/24HOURS": "MG/24HR",
            "SECONDS": "SEC"
                       }
    conversion_table = {
            50889: {"MG/DL": ["MG/L", 10.0]},
            50916: {"NG/ML": ["UG/DL", 0.1]},
            50926: {"MIU/L": ["MIU/ML", 0.001]},
            50958: {"MIU/L": ["MIU/ML", 0.001]},
            50974: {"UG/L": ["NG/ML", 1.0]},
            50989: {"NG/DL": ["PG/ML", 10.0]},
            51514: {"EU/DL": ["MG/DL", 0.0016605402]},
            50964: {"MOSM/L": ["MOSM/KG", 1.0]},
            50993: {"UU/ML": ["UIU/ML", 1.0]},
            51127: {"#/CU MM": ["#/UL", 1.0]},
            51128: {"#/CU MM": ["#/UL", 1.0]},
            3451: {"KG": ["CM", 1.0]},
            3723: {"CM": ["KG", 1.0]},
                       }

    row["VALUEUOM"] = row["VALUEUOM"].strip().upper()
    try:
        row["VALUEUOM"] = regularize_table[row["VALUEUOM"]]
    except KeyError:
        pass

    try:
        row["VALUEUOM"] = conversion_table[int(row["ITEMID"])][row["VALUEUOM"]][0]
        row["VALUENUM"] *= conversion_table[int(row["ITEMID"])][row["VALUEUOM"]][1]
    except KeyError:
        pass

    if pbar is not None:
        pbar.update(1)

    return row
    

def main(args):

    # files can be downloaded from https://mimic.physionet.org/gettingstarted/dbsetup/
    med_file = os.path.join(args.data_path, 'PRESCRIPTIONS.csv')
    diag_file = os.path.join(args.data_path, 'DIAGNOSES_ICD.csv')
    proc_file = os.path.join(args.data_path, 'PROCEDURES_ICD.csv')

    # files for pretrain
    lab_file = os.path.join(args.data_path, 'LABEVENTS.csv')
    chart_file = os.path.join(args.data_path, 'CHARTEVENTS.csv')
    item_file = os.path.join(args.data_path, 'D_ITEMS.csv')
    labitem_file = os.path.join(args.data_path, 'D_LABITEMS.csv')


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

    def get_item():
        item_pd = pd.read_csv(item_file)
        labitem_pd = pd.read_csv(labitem_file)
        item_pd.drop(columns=['ROW_ID', "ABBREVIATION", "CATEGORY", "DBSOURCE", "LINKSTO", "UNITNAME", "PARAM_TYPE", "CONCEPTID"], inplace=True)
        labitem_pd.drop(columns=['ROW_ID', "CATEGORY", "LOINC_CODE", "FLUID"], inplace=True)
        labitem_pd.drop_duplicates(inplace=True)
        item_pd.drop_duplicates(inplace=True)
        item_pd = pd.concat([item_pd, labitem_pd])
        item_pd.reset_index(drop=True, inplace=True)

        return item_pd

    def get_lab():
        lab_pd = read_huge_csv(lab_file, nrows=5e6 if args.run_local else None,
                               low_memory=False)
        lab_pd.drop(columns=['ROW_ID', "VALUE"], inplace=True)
        lab_pd["VALUEUOM"].fillna("", inplace=True)

        chunksize = int(1e5)
        values = lab_pd[["ITEMID", "VALUENUM", "VALUEUOM"]]
        value_chunks = [values[i: i+chunksize] for i in range(0, len(values), chunksize)]
        for i, value_chunk in tqdm(enumerate(value_chunks), total=len(value_chunks), desc="regularizing unit"):
            lab_pd.loc[i*chunksize: (i+1)*chunksize, ["ITEMID", "VALUENUM", "VALUEUOM"]] = value_chunk.apply(regularize_unit, axis=1)
        #with tqdm(total=len(values), desc="regularizing unit") as pbar:
        #    regularize_func = partial(regularize_unit, pbar=pbar)
        #    lab_pd[["ITEMID", "VALUENUM", "VALUEUOM"]] = lab_pd[["ITEMID", "VALUENUM", "VALUEUOM"]].apply(regularize_func, axis=1)
        #lab_pd["VALUEUOM"] = lab_pd["VALUEUOM"].map(lambda x: x.strip().upper())
        lab_pd.drop_duplicates(inplace=True)
        lab_pd_no_flag = lab_pd.drop(columns=["FLAG"])
        lab_pd_no_flag.dropna(inplace=True)
        lab_pd = pd.merge(lab_pd, lab_pd_no_flag, on=["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM", "VALUEUOM"], how="inner")
        lab_pd.fillna("normal", inplace=True)
        lab_pd['HADM_ID'] = lab_pd['HADM_ID'].astype('int64')
        lab_pd['CHARTTIME'] = pd.to_datetime(lab_pd['CHARTTIME'], format='%Y-%m-%d %H:%M:%S')    
        lab_pd.drop_duplicates(inplace=True)
        #lab_pd.sort_values(by=['SUBJECT_ID','HADM_ID'], inplace=True)
        lab_pd = lab_pd.reset_index(drop=True)

        return lab_pd

    def get_chart():
        chart_pd = read_huge_csv(chart_file, nrows=5e6 if args.run_local else None, 
                                 low_memory=False)
        chart_pd.drop(columns=["ROW_ID", "ICUSTAY_ID", "STORETIME", "CGID", "VALUE", "RESULTSTATUS"], inplace=True)
        chart_pd["VALUEUOM"].fillna("", inplace=True)

        chunksize = int(1e5)
        values = chart_pd[["ITEMID", "VALUENUM", "VALUEUOM"]]
        value_chunks = [values[i: i+chunksize] for i in range(0, len(values), chunksize)]
        for i, value_chunk in tqdm(enumerate(value_chunks), total=len(value_chunks), desc="regularizing unit"):
            chart_pd.loc[i*chunksize: (i+1)*chunksize, ["ITEMID", "VALUENUM", "VALUEUOM"]] = value_chunk.apply(regularize_unit, axis=1)
        chart_pd.drop_duplicates(inplace=True)

        # remove records with warning and error
        chart_pd.drop(index=chart_pd[chart_pd['WARNING'] == 1].index, inplace=True)
        chart_pd.drop(index=chart_pd[chart_pd['ERROR'] == 1].index, inplace=True)
        chart_pd.drop(columns=["WARNING", "ERROR"], inplace=True)
        chart_pd.drop_duplicates(inplace=True)

        chart_pd.drop(index=chart_pd[chart_pd['STOPPED'] == "D/C'd'"].index, inplace=True)
        chart_pd.drop(columns=["STOPPED"], inplace=True)
        chart_pd.dropna(inplace=True)
        chart_pd.drop_duplicates(inplace=True)

        chart_pd['CHARTTIME'] = pd.to_datetime(chart_pd['CHARTTIME'], format='%Y-%m-%d %H:%M:%S')    
        #chart_pd.sort_values(by=['SUBJECT_ID','HADM_ID'], inplace=True)
        chart_pd = chart_pd.reset_index(drop=True)
        return chart_pd

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
    
    if args.pretrain:

        med_pd = get_med()
        med_pd.drop(columns=["ICUSTAY_ID"], inplace=True)
        med_pd.rename(columns={"STARTDATE": "DATETIME"}, inplace=True)
        simple_statistics(med_pd, typ="med", description="Get all medications")

        diag_pd = get_diag()
        simple_statistics(diag_pd, typ="diag", description="Get all diagnoses")

        proc_pd = get_proc()
        simple_statistics(proc_pd, typ="proc", description="Get all procedures")

        item_pd = get_item()

        dfs = [med_pd, diag_pd, proc_pd]

        lab_pd = get_lab()
        lab_pd.rename(columns={"CHARTTIME": "DATETIME"}, inplace=True)

        chart_pd = get_chart()
        chart_pd.rename(columns={"CHARTTIME": "DATETIME"}, inplace=True)

        data = pd.concat([med_pd, diag_pd, proc_pd, lab_pd, chart_pd])
        data.sort_values(by=["SUBJECT_ID", "HADM_ID", "DATETIME"], inplace=True)
        data.reset_index(inplace=True)

        # handle units

        # drop units not understood
        data.drop(index=data.loc[(data["ITEMID"] == 113.0) & (data["VALUEUOM"] == "%")].index, inplace=True)
        data.drop(index=data.loc[(data["ITEMID"] == 50980.0) & (data["VALUEUOM"] == "I.U.")].index, inplace=True)
        item_ids = list(set(data["ITEMID"].dropna()))
        for item_id in item_ids:
            item_df = data[data["ITEMID"] == item_id]
            unit_set = set(item_df["VALUEUOM"])
            # replace empty unit with default unit 
            if "" in unit_set and len(unit_set) == 2:
                unit_set.remove("")
                default_unit = unit_set.pop()
                empty_index = data.loc[(data["ITEMID"] == item_id) & (data["VALUEUOM"] == "")].index
                data.loc[empty_index, ["VALUEUOM"]] = default_unit
            # print items with multiple unit
            if len(unit_set) > 1:
                print(item_id, item_pd[item_pd["ITEMID"] == int(item_id)]["LABEL"].to_list()[0], unit_set)
                for unit in unit_set:
                    unit_count = len(item_df[item_df["VALUEUOM"] == unit])
                    print("#count {}: {}".format(unit, unit_count))
        data.to_csv("{}.csv".format(args.output))
        ipdb.set_trace()
        pass
    else:
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
