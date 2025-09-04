import pandas as pd
import torch


admissions = pd.read_csv('/data1/wanghao/mimic-iii-clinical-database-1.4/ADMISSIONS.csv')
admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
admissions['DISCHTIME'] = pd.to_datetime(admissions['DISCHTIME'])
def generate_samples():
    patients = admissions['SUBJECT_ID'].unique()
    samples_multiwindow = []
    samples_onewindow = []
    samples = []

    def generate_sample(patient_id):
        # admissions of current patient
        patient_admissions = admissions[admissions['SUBJECT_ID'] == patient_id]
        patient_samples_multiwindow = []
        patient_samples_onewindow = []
        for index, row in patient_admissions.iterrows():
            admit_time = row['ADMITTIME']
            sample = []
            for i in range(6):
                start_time = admit_time - pd.Timedelta(days=20 * (6 - i))
                end_time = start_time + pd.Timedelta(days=20)
                # get window admissions
                window_admission = patient_admissions[(patient_admissions['ADMITTIME'] >= start_time) & (patient_admissions['ADMITTIME'] < end_time)]
                visit = []
                for _, wr in window_admission.iterrows():
                    visit_id = wr['HADM_ID']
                    id = (patient_id, visit_id)
                    visit.append(id)
                sample.append(visit)
            sample[5].append((patient_id, row['HADM_ID']))

            # filter out sample with only one window within 120 days
            for i in range(6):
                if i < 5 and len(sample[i]) > 0:
                    patient_samples_multiwindow.append(sample)
                    break
                if i == 5:
                    patient_samples_onewindow.append(sample)
        
        return patient_samples_multiwindow, patient_samples_onewindow

    for patient_id in tqdm(patients, total=len(patients), desc="generating samples"):
        # 生成每个admission的时序特征
        patient_samples_multiwindow, patient_samples_onewindow = generate_sample(patient_id)
        if len(patient_samples_multiwindow) > 0:
            samples_multiwindow += patient_samples_multiwindow
        if len(patient_samples_onewindow) > 0:
            samples_onewindow += patient_samples_onewindow
    
    samples += samples_multiwindow
    print(len(samples))

    return samples

def extract_admission_diagnoses(samples):
    diagnoses = pd.read_csv('/data1/wanghao/mimic-iii-clinical-database-1.4/DIAGNOSES_ICD.csv')
    admission_diagnoses = {}
    for si, sample in tqdm(enumerate(samples), total=len(samples), desc="extracting admission diagnoses"):
        last_visit = sample[-1][-1]
        patient_id, visit_id = last_visit
        visit_dia = diagnoses[(diagnoses['SUBJECT_ID'] == patient_id) & (diagnoses['HADM_ID'] == visit_id)]
        for di, dia in visit_dia.iterrows():
            if dia['ICD9_CODE'] in admission_diagnoses:
                admission_diagnoses[dia['ICD9_CODE']] += 1
            else:
                admission_diagnoses[dia['ICD9_CODE']] = 1
    sorted_admission_diagnoses = sorted(admission_diagnoses.items(), key=lambda item: item[1], reverse=True)
    admission_diagnoses = [item[0] for item in sorted_admission_diagnoses[:200]]
    return admission_diagnoses

def generate_code_map(samples, admission_diagnoses):
    '''
    get dictionary of medical code
    Parameters:
    - samples: List of patient visit data
    - admission_diagnoses: List of diagnosis codes that could be used as labels
    
    Returns:
    - code_map: Dictionary mapping codes to indices
    '''
    prescriptions = pd.read_csv('/data1/wanghao/mimic-iii-clinical-database-1.4/PRESCRIPTIONS.csv')
    prescriptions['ENDDATE'] = pd.to_datetime(prescriptions['ENDDATE'])
    procedures = pd.read_csv('/data1/wanghao/mimic-iii-clinical-database-1.4/PROCEDURES_ICD.csv')
    diagnoses = pd.read_csv('/data1/wanghao/mimic-iii-clinical-database-1.4/DIAGNOSES_ICD.csv')
    dia_map = {}
    pro_map = {}
    drug_map = {}
    for sample in tqdm(samples, total=len(samples), desc="generating code map"):
        for window in sample:
            for visit in window:
                patient_id, visit_id = visit

                visit_dia = diagnoses[(diagnoses['SUBJECT_ID'] == patient_id) & (diagnoses['HADM_ID'] == visit_id)]
                visit_dia = list(visit_dia['ICD9_CODE'])
                for dia in visit_dia:
                    if dia not in dia_map:
                        dia_map[dia] = 1
                    else:
                        dia_map[dia] += 1
                
                pro_dia = procedures[(procedures['SUBJECT_ID'] == patient_id) & (procedures['HADM_ID'] == visit_id)]
                pro_dia = list(pro_dia['ICD9_CODE'])
                for pro in pro_dia:
                    if pro not in pro_map:
                        pro_map[pro] = 1
                    else:
                        pro_map[pro] += 1

                discharge_time = admissions[(admissions['SUBJECT_ID'] == patient_id) & (admissions['HADM_ID'] == visit_id)]['DISCHTIME'].values[0]
                visit_drug = prescriptions[(prescriptions['SUBJECT_ID'] == patient_id) & (prescriptions['HADM_ID'] == visit_id) & (prescriptions['ENDDATE'].isna() | (prescriptions['ENDDATE'] < discharge_time)) & (prescriptions['DRUG_TYPE'] == 'MAIN')]
                visit_drug = list(visit_drug['DRUG'])
                for drug in visit_drug:
                    if drug not in drug_map:
                        drug_map[drug] = 1
                    else:
                        drug_map[drug] += 1

    print(f'dia_map:{len(dia_map)}')
    print(f'pro_map:{len(pro_map)}')
    print(f'drug_map:{len(drug_map)}')
    
    # ------------ FEATURE SELECTION IMPROVEMENT START ------------
    # 1. Sort all maps by frequency
    sorted_dia = sorted(dia_map.items(), key=lambda item: item[1], reverse=True)
    sorted_pro = sorted(pro_map.items(), key=lambda item: item[1], reverse=True)
    sorted_drug = sorted(drug_map.items(), key=lambda item: item[1], reverse=True)
    
    # 2. Determine the number of features to select from each category
    # Option 1: Proportional selection (50 diagnoses, 20 procedures, 30 drugs = 100 total)
    dia_count = 200
    pro_count = 120
    drug_count = 180
    
    # Option 2: Alternative - prioritize by clinical importance
    # dia_count = 60  # Prioritize diagnoses
    # pro_count = 15
    # drug_count = 25
    
    # 3. Select top features by frequency
    dia_map = dict(sorted_dia[:dia_count])
    pro_map = dict(sorted_pro[:pro_count])
    drug_map = dict(sorted_drug[:drug_count])
    
    # 4. Create a unified code map with the selected features
    code_map = {}
    # Add diagnosis codes
    for code, _ in dia_map.items():
        if code not in code_map:
            code_map[code] = len(code_map)
    # Add procedure codes
    for code, _ in pro_map.items():
        if code not in code_map:
            code_map[code] = len(code_map)
    # Add drug codes
    for code, _ in drug_map.items():
        if code not in code_map:
            code_map[code] = len(code_map)
    
    # 5. Ensure admission diagnoses are included (these might be crucial for the task)
    # If needed, you can limit this to only add admission diagnoses up to a certain total count
    admission_diagnoses_added = 0
    new_admission_diagnoses = []
    for dia in admission_diagnoses:
        if dia not in code_map and admission_diagnoses_added < 50:
            admission_diagnoses_added += 1
            new_admission_diagnoses.append(dia)
            code_map[dia] = len(code_map)
    
    print(f'code_map:{len(code_map)} (including {admission_diagnoses_added} additional admission diagnoses)')
    
    return code_map, new_admission_diagnoses

def generate_one_hot(samples, code_map, admission_diagnoses):
    '''
    generate multi hot vector
    '''
    prescriptions = pd.read_csv('/data1/wanghao/mimic-iii-clinical-database-1.4/PRESCRIPTIONS.csv')
    prescriptions['ENDDATE'] = pd.to_datetime(prescriptions['ENDDATE'])
    procedures = pd.read_csv('/data1/wanghao/mimic-iii-clinical-database-1.4/PROCEDURES_ICD.csv')
    diagnoses = pd.read_csv('/data1/wanghao/mimic-iii-clinical-database-1.4/DIAGNOSES_ICD.csv')
    features_one_hot = []
    label_one_hot = []
    max_feat_num = 0
    for si, sample in tqdm(enumerate(samples), total=len(samples), desc="generating one hot vector"):
        sample_features = []
        sample_label = torch.zeros(size=(1, len(admission_diagnoses)))
        # generating features
        for wi, window in enumerate(sample):
            window_features = torch.zeros(size=(1, len(code_map)))
            for vi, visit in enumerate(window):
                patient_id, visit_id = visit
                # diagnoses
                if (wi != len(sample) - 1) or (vi != len(window) - 1): # not last visit
                    visit_dia = diagnoses[(diagnoses['SUBJECT_ID'] == patient_id) & (diagnoses['HADM_ID'] == visit_id)]
                    for di, dia in visit_dia.iterrows():
                        if dia['ICD9_CODE'] in code_map:
                            window_features[0][code_map[dia['ICD9_CODE']]] += 1

                # procedures
                visit_pro = procedures[(procedures['SUBJECT_ID'] == patient_id) & (procedures['HADM_ID'] == visit_id)]
                for pi, pro in visit_pro.iterrows():
                    if pro['ICD9_CODE'] in code_map:
                        window_features[0][code_map[pro['ICD9_CODE']]] += 1

                # medications
                visit_drug = prescriptions[(prescriptions['SUBJECT_ID'] == patient_id) & (prescriptions['HADM_ID'] == visit_id)]
                for di, drug in visit_drug.iterrows():
                    if drug['DRUG'] in code_map:
                        window_features[0][code_map[drug['DRUG']]] += 1
            feat_num = torch.sum(window_features, dim = -1)
            feat_num = feat_num.item()
            if feat_num > max_feat_num:
                max_feat_num = feat_num
            sample_features.append(window_features)
        # generating labels
        last_visit = sample[-1][-1]
        patient_id, visit_id = last_visit
        visit_dia = diagnoses[(diagnoses['SUBJECT_ID'] == patient_id) & (diagnoses['HADM_ID'] == visit_id)]
        for di, dia in visit_dia.iterrows():
            if dia['ICD9_CODE'] in code_map and dia['ICD9_CODE'] in admission_diagnoses:
                sample_label[0][admission_diagnoses.index(dia['ICD9_CODE'])] = 1

        features_one_hot.append(sample_features)
        label_one_hot.append(sample_label)
    assert len(features_one_hot) == len(label_one_hot)
    print(max_feat_num)
    torch.save(features_one_hot, './features_one_hot.pt')
    torch.save(label_one_hot, './label_one_hot.pt')