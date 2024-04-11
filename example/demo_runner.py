import os
import dill
import pickle
import pandas as pd
import numpy as np
from pytrial.data.patient_data import TabularPatientBase
from promptehr import PromptEHR
import ast
os.chdir('../')
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"




from pytrial.tasks.trial_simulation.data import SequencePatient
from pytrial.data.demo_data import load_mimic_ehr_sequence

# from promptehr import load_demo_data

# load pytrial demodata, supported by PyTrial package to load the demo EHR data
# from pytrial.data.demo_data import load_mimic_ehr_sequence
# from pytrial.tasks.trial_simulation.data import SequencePatient

# input_dir = r'C:\Users\zhong\PycharmProjects\PromptEHR\demo_data\demo_patient_sequence\ehr\'

def see(input_dir, shortICD = True, breast = False):
    assert shortICD != breast
    if shortICD:
        train_data = pd.read_csv(input_dir + 'train' + '_3digmimic.csv')
        test_data = pd.read_csv(input_dir + 'test' + '_3digmimic.csv')
        val_data = pd.read_csv(input_dir + 'val' + '_3digmimic.csv')

        diag_voc = pd.read_csv(input_dir + 'diagnosis_to_int_mapping_3dig.csv')
        drug_voc = pd.read_csv(input_dir + 'drug_to_int_mapping_3dig.csv')
        lab_voc = pd.read_csv(input_dir + 'lab_to_int_mapping_3dig.csv')
        proc_voc = pd.read_csv(input_dir + 'proc_to_int_mapping_3dig.csv')

        total_patents = len(train_data) + len(test_data) + len(val_data)

        print('3dig total_patents', total_patents)
        print('3dig total diagnosis codes', len(diag_voc))
        print('3dig total drug codes', len(drug_voc))
        print('3dig total lab codes', len(lab_voc))
        print('3dig total proc codes', len(proc_voc))
    elif breast:
        train_data = pd.read_csv(input_dir + 'train' + '_3digbreast.csv')
        test_data = pd.read_csv(input_dir + 'test' + '_3digbreast.csv')
        val_data = pd.read_csv(input_dir + 'val' + '_3digbreast.csv')

        diag_voc = pd.read_csv(input_dir + 'ae_to_int_mapping_3dig.csv')
        drug_voc = pd.read_csv(input_dir + 'drug_to_int_mapping_3dig.csv')
        lab_voc = pd.read_csv(input_dir + 'lab_to_int_mapping_3dig.csv')
        proc_voc = pd.read_csv(input_dir + 'proc_to_int_mapping_3dig.csv')
        total_patents = len(train_data) + len(test_data) + len(val_data)
        print('3dig total_patents', total_patents)
        print('3dig total diagnosis codes', len(diag_voc))
        print('3dig total drug codes', len(drug_voc))
        print('3dig total lab codes', len(lab_voc))
        print('3dig total proc codes', len(proc_voc))

    else:

        train_data = pd.read_csv(input_dir + 'train' + '_5digmimic.csv')
        test_data = pd.read_csv(input_dir + 'test' + '_5digmimic.csv')
        val_data = pd.read_csv(input_dir + 'val' + '_5digmimic.csv')

        diag_voc = pd.read_csv(input_dir + 'diagnosis_to_int_mapping_5dig.csv')
        drug_voc = pd.read_csv(input_dir + 'drug_to_int_mapping_5dig.csv')
        lab_voc = pd.read_csv(input_dir + 'lab_to_int_mapping_5dig.csv')
        proc_voc = pd.read_csv(input_dir + 'proc_to_int_mapping_5dig.csv')

        total_patents = len(train_data) + len(test_data) + len(val_data)

        print('5dig total_patents', total_patents)
        print('5dig total diagnosis codes', len(diag_voc))
        print('5dig total drug codes', len(drug_voc))
        print('5dig total lab codes', len(lab_voc))
        print('5dig total proc codes', len(proc_voc))


# see the input format
# demo = load_mimic_ehr_sequence(input_dir=input_dir, n_sample=100)
#
# seqdata = SequencePatient(data={'v': demo['visit'], 'y': demo['mortality'], 'x': demo['feature'], },
#                           metadata={
#                               'visit': {'mode': 'dense'},
#                               'label': {'mode': 'tensor'},
#                               'voc': demo['voc'],
#                               'max_visit': 20,
#                           }
#                           )

# def loadingData(name):
#     data = pd.read_csv(input_dir + name + '_mimic.csv')
#     visit = data['visit'].apply(lambda x: ast.literal_eval(x)).tolist()
#     for i in range(len(visit)):
#         for j in range(len(visit[i])):
#             visit[i][j] = [visit[i][j]]
#     voc = pd.read_csv(input_dir + 'code_to_int_mapping.csv')
#     feature = data.drop(columns='visit')
#     label = feature['MORTALITY'].values
#     x = feature[['AGE','GENDER','ETHNICITY']]
#     tabx = TabularPatientBase(x)
#     x = tabx.df.values
#
#     n_num_feature = 1
#     cat_cardinalities = []
#     for i in range(n_num_feature, x.shape[1]):
#         cat_cardinalities.append(len(list(set(x[:,i]))))
#
#     demo = {
#             'visit':visit,
#             'voc':voc,
#             'order':['diag'],
#             'mortality':label,
#             'feature':x,
#             'n_num_feature':n_num_feature,
#             'cat_cardinalities':cat_cardinalities,
#             }
#
#     seqdata = SequencePatient(data={'v':demo['visit'], 'y':demo['mortality'], 'x':demo['feature'],},
#         metadata={
#             'visit':{'mode':'dense'},
#             'label':{'mode':'tensor'},
#             'voc':demo['voc'],
#             'max_visit':20,
#             }
#         )
#     return seqdata, demo
#
# print('visit', demo['visit'][4]) # a list of visit events
# print('mortality', demo['mortality'][4]) # array of labels
# print('feature', demo['feature'][4]) # array of patient baseline features
# print('voc', demo['voc']) # dict of dicts containing the mapping from index to the original event names
# print('order', demo['order']) # a list of three types of code
# print('n_num_feature', demo['n_num_feature']) # int: a number of patient's numerical features
# print('cat_cardinalities', demo['cat_cardinalities']) # list: a list of cardinalities of patient's categorical features
#
# train_seq, train_demo = loadingData('train')
# # toy_seq, toy_demo = loadingData('toy')
# test_seq, test_demo = loadingData('test')
# val_seq, val_demo = loadingData('val')
#
# # fit the model
# model = PromptEHR(
#     code_type=test_demo['order'],
#     n_num_feature=test_demo['n_num_feature'],
#     cat_cardinalities=test_demo['cat_cardinalities'],
#     num_worker=0,
#     eval_step=1,
#     epoch=1,
#     device=[1,2],
# )
# model.fit(
#     train_data=train_seq,
#     val_data=test_seq,
# )


########################################################
#4 modalities

def loadingData(shortICD, name, input_dir, subset, breast, mimic4, eicu):
    if mimic4:
        data = pd.read_csv(input_dir + name + '_3digmimic4.csv')
        diag_voc = pd.read_csv(input_dir + 'diagnosis_to_int_mapping_mimic4.csv')
        drug_voc = pd.read_csv(input_dir + 'drug_to_int_mapping_mimic4.csv')
        lab_voc = pd.read_csv(input_dir + 'lab_to_int_mapping_mimic4.csv')
        proc_voc = pd.read_csv(input_dir + 'proc_to_int_mapping_mimic4.csv')
    elif eicu:
        data = pd.read_csv(input_dir + name + '_3digeicu.csv')
        diag_voc = pd.read_csv(input_dir + 'diagnosis_to_int_mapping_3digeicu.csv')
        drug_voc = pd.read_csv(input_dir + 'drug_to_int_mapping_3digeicu.csv')
        lab_voc = pd.read_csv(input_dir + 'lab_to_int_mapping_3digeicu.csv')
        proc_voc = pd.read_csv(input_dir + 'proc_to_int_mapping_3digeicu.csv')
    elif breast:
        data = pd.read_csv(input_dir + name + '_3digbreast.csv')
        diag_voc = pd.read_csv(input_dir + 'ae_to_int_mapping.csv')
        drug_voc = pd.read_csv(input_dir + 'drug_to_int_mapping.csv')
        lab_voc = pd.read_csv(input_dir + 'lab_to_int_mapping.csv')
        proc_voc = pd.read_csv(input_dir + 'proc_to_int_mapping.csv')
    elif shortICD and not subset:
        data = pd.read_csv(input_dir + name + '_3digmimic.csv')
        diag_voc = pd.read_csv(input_dir + 'diagnosis_to_int_mapping_3dig.csv')
        drug_voc = pd.read_csv(input_dir + 'drug_to_int_mapping_3dig.csv')
        lab_voc = pd.read_csv(input_dir + 'lab_to_int_mapping_3dig.csv')
        proc_voc = pd.read_csv(input_dir + 'proc_to_int_mapping_3dig.csv')
    elif not shortICD and not subset:
        data = pd.read_csv(input_dir + name + '_5digmimic.csv')
        diag_voc = pd.read_csv(input_dir + 'diagnosis_to_int_mapping_5dig.csv')
        drug_voc = pd.read_csv(input_dir + 'drug_to_int_mapping_5dig.csv')
        lab_voc = pd.read_csv(input_dir + 'lab_to_int_mapping_5dig.csv')
        proc_voc = pd.read_csv(input_dir + 'proc_to_int_mapping_5dig.csv')
    elif shortICD and subset:
        data = pd.read_csv(input_dir + name + '_3digmimic3_subset.csv')
        diag_voc = pd.read_csv(input_dir + 'diagnosis_to_int_mapping_3dig.csv')
        drug_voc = pd.read_csv(input_dir + 'drug_to_int_mapping_3dig.csv')
        lab_voc = pd.read_csv(input_dir + 'lab_to_int_mapping_3dig.csv')
        proc_voc = pd.read_csv(input_dir + 'proc_to_int_mapping_3dig.csv')
    elif not shortICD and subset:
        data = pd.read_csv(input_dir + name + '_5digmimic3_subset.csv')
        diag_voc = pd.read_csv(input_dir + 'diagnosis_to_int_mapping_5dig.csv')
        drug_voc = pd.read_csv(input_dir + 'drug_to_int_mapping_5dig.csv')
        lab_voc = pd.read_csv(input_dir + 'lab_to_int_mapping_5dig.csv')
        proc_voc = pd.read_csv(input_dir + 'proc_to_int_mapping_5dig.csv')

    diag = data['DIAGNOSES_int'].apply(lambda x: ast.literal_eval(x)).tolist()
    drug = data['DRG_CODE_int'].apply(lambda x: ast.literal_eval(x)).tolist()
    lab = data['LAB_ITEM_int'].apply(lambda x: ast.literal_eval(x)).tolist()
    proc = data['PROC_ITEM_int'].apply(lambda x: ast.literal_eval(x)).tolist()
    #

    visit = [
    [  # For each patient
        [diag_visit, drug_visit, lab_visit, proc_visit]  # Nest each type of data in a separate sublist for each visit
        for diag_visit, drug_visit, lab_visit, proc_visit in zip(diag_patient, drug_patient, lab_patient, proc_patient)
    ]
    for diag_patient, drug_patient, lab_patient, proc_patient in zip(diag, drug, lab, proc)
]

    # visit = data['visit'].apply(lambda x: ast.literal_eval(x)).tolist()
    # for i in range(len(visit)):
    #     for j in range(len(visit[i])):
    #         visit[i][j] = [visit[i][j]]

    #putting all vocs together
    voc = {'diag':diag_voc, 'drug':drug_voc, 'lab':lab_voc, 'proc':proc_voc}
    order = ['diag', 'drug', 'lab', 'proc']
    #select AGE,GENDER,ETHNICITY,MORTALITY columns from data
    if mimic4 or eicu:
        feature = data[['AGE', 'GENDER', 'MORTALITY']]
    else:
        feature = data[['AGE', 'GENDER', 'ETHNICITY', 'MORTALITY']]
    label = feature['MORTALITY'].values
    if mimic4 or eicu:
        x = feature[['AGE','GENDER']]
    else:
        x = feature[['AGE','GENDER','ETHNICITY']]
    tabx = TabularPatientBase(x)
    x = tabx.df.values

    n_num_feature = 1
    cat_cardinalities = []
    for i in range(n_num_feature, x.shape[1]):
        cat_cardinalities.append(len(list(set(x[:,i]))))

    demo = {
            'visit':visit,
            'voc':voc,
            'order': order,
            'mortality':label,
            'feature':x,
            'n_num_feature':n_num_feature,
            'cat_cardinalities':cat_cardinalities,
            }

    seqdata = SequencePatient(data={'v':demo['visit'], 'y':demo['mortality'], 'x':demo['feature'],},
        metadata={
            'visit':{'mode':'dense'},
            'label':{'mode':'tensor'},
            'voc':demo['voc'],
            'max_visit':20,
            }
        )
    return seqdata, demo

def main(input_dir, shortICD, breast, mimic4, eicu):


    train_seq, train_demo = loadingData(shortICD,'train',input_dir, False, breast, mimic4, eicu)
    toy_seq, toy_demo = loadingData(shortICD,'toy',input_dir, False, breast, mimic4, eicu)
    test_seq, test_demo = loadingData(shortICD,'test',input_dir, False, breast, mimic4, eicu)
    val_seq, val_demo = loadingData(shortICD,'val',input_dir, False, breast, mimic4, eicu)

    print('visit', test_demo['visit'][0]) # a list of visit events
    print('mortality', test_demo['mortality'][0]) # array of labels
    print('feature', test_demo['feature'][0]) # array of patient baseline features
    print('voc', test_demo['voc']) # dict of dicts containing the mapping from index to the original event names
    print('order', test_demo['order']) # a list of three types of code
    print('n_num_feature', test_demo['n_num_feature']) # int: a number of patient's numerical features
    print('cat_cardinalities', test_demo['cat_cardinalities']) # list: a list of cardinalities of patient's categorical features

    # fit the model
    model = PromptEHR(
        code_type=test_demo['order'],
        n_num_feature=test_demo['n_num_feature'],
        cat_cardinalities=test_demo['cat_cardinalities'],
        num_worker=0,
        eval_step=1e6,
        epoch=1,
        device=[1,2],
        shortICD=shortICD,
        eval_batch_size = 16
    )
    model.fit(
        train_data=train_seq,
        val_data=val_seq,
    )
    if mimic4:
        model.save_model('C:\\Users\\yfz5556\\PycharmProjects\\PromptEHR\\simulation_mimic4')
    elif eicu:
        model.save_model('C:\\Users\\yfz5556\\PycharmProjects\\PromptEHR\\simulation_eicu')
    elif breast:
        model.save_model('C:\\Users\\yfz5556\\PycharmProjects\\PromptEHR\\simulation_breast')
    elif shortICD:
        model.save_model('C:\\Users\\yfz5556\\PycharmProjects\\PromptEHR\\simulation_shortICD')
    else:
        model.save_model('C:\\Users\\yfz5556\\PycharmProjects\\PromptEHR\\simulation_longICD')

    model.evaluate(test_seq)

def predict(input_dir, shortICD, subset, breast, mimic4, eicu):
    model = PromptEHR()
    if mimic4:
        model.load_model('C:\\Users\\yfz5556\\PycharmProjects\\PromptEHR\\simulation_mimic4')
    elif eicu:
        model.load_model('C:\\Users\\yfz5556\\PycharmProjects\\PromptEHR\\simulation_eicu')
    elif breast:
        model.load_model('C:\\Users\\yfz5556\\PycharmProjects\\PromptEHR\\simulation_breast')
    elif shortICD:
        model.load_model('C:\\Users\\yfz5556\\PycharmProjects\\PromptEHR\\simulation_shortICD')
    else:
        model.load_model('C:\\Users\\yfz5556\\PycharmProjects\\PromptEHR\\simulation_longICD')
    tests_subset_seq, tests_subset_demo = loadingData(shortICD,'train',input_dir, subset, breast, mimic4)
    return_res = model.predict(tests_subset_seq, n=59651)
    #write the prediction results to csv
    if mimic4:
        for key, value in return_res.items():
            if isinstance(value, np.ndarray) and value.ndim > 1:
                value = value.tolist()
            # If the value is a scalar, convert it to a 1D list
            elif not isinstance(value, (list, pd.Series, np.ndarray)):
                value = [value]

            # Convert the value to a DataFrame
            df = pd.DataFrame({key: value})

            # Construct the file path
            file_path = f'C:\\Users\\yfz5556\\PycharmProjects\\PromptEHR\\simulation_mimic4\\{key}.csv'

            # Save the DataFrame to a CSV file
            df.to_csv(file_path, index=False)
    elif eicu:
        for key, value in return_res.items():
            if isinstance(value, np.ndarray) and value.ndim > 1:
                value = value.tolist()
            # If the value is a scalar, convert it to a 1D list
            elif not isinstance(value, (list, pd.Series, np.ndarray)):
                value = [value]

            # Convert the value to a DataFrame
            df = pd.DataFrame({key: value})

            # Construct the file path
            file_path = f'C:\\Users\\yfz5556\\PycharmProjects\\PromptEHR\\simulation_eicu\\{key}.csv'

            # Save the DataFrame to a CSV file
            df.to_csv(file_path, index=False)
    elif breast:
        for key, value in return_res.items():
            if isinstance(value, np.ndarray) and value.ndim > 1:
                value = value.tolist()
            # If the value is a scalar, convert it to a 1D list
            elif not isinstance(value, (list, pd.Series, np.ndarray)):
                value = [value]

            # Convert the value to a DataFrame
            df = pd.DataFrame({key: value})

            # Construct the file path
            file_path = f'C:\\Users\\yfz5556\\PycharmProjects\\PromptEHR\\simulation_breast\\{key}.csv'

            # Save the DataFrame to a CSV file
            df.to_csv(file_path, index=False)
    elif shortICD and subset:
        for key, value in return_res.items():
            if isinstance(value, np.ndarray) and value.ndim > 1:
                value = value.tolist()
            # If the value is a scalar, convert it to a 1D list
            elif not isinstance(value, (list, pd.Series, np.ndarray)):
                value = [value]

            # Convert the value to a DataFrame
            df = pd.DataFrame({key: value})

            # Construct the file path
            file_path = f'C:\\Users\\yfz5556\\PycharmProjects\\PromptEHR\\simulation_shortICD\\{key}_subset.csv'

            # Save the DataFrame to a CSV file
            df.to_csv(file_path, index=False)
    elif shortICD and not subset:
        for key, value in return_res.items():
            if isinstance(value, np.ndarray) and value.ndim > 1:
                value = value.tolist()
            # If the value is a scalar, convert it to a 1D list
            elif not isinstance(value, (list, pd.Series, np.ndarray)):
                value = [value]

            # Convert the value to a DataFrame
            df = pd.DataFrame({key: value})

            # Construct the file path
            file_path = f'C:\\Users\\yfz5556\\PycharmProjects\\PromptEHR\\simulation_shortICD\\{key}.csv'

            # Save the DataFrame to a CSV file
            df.to_csv(file_path, index=False)
    elif not shortICD and subset:
        for key, value in return_res.items():
            if isinstance(value, np.ndarray) and value.ndim > 1:
                value = value.tolist()
            # If the value is a scalar, convert it to a 1D list
            elif not isinstance(value, (list, pd.Series, np.ndarray)):
                value = [value]

            # Convert the value to a DataFrame
            df = pd.DataFrame({key: value})

            # Construct the file path
            file_path = f'C:\\Users\\yfz5556\\PycharmProjects\\PromptEHR\\simulation_longICD\\{key}_subset.csv'

            # Save the DataFrame to a CSV file
            df.to_csv(file_path, index=False)
    elif not shortICD and not subset:
        for key, value in return_res.items():
            if isinstance(value, np.ndarray) and value.ndim > 1:
                value = value.tolist()
            # If the value is a scalar, convert it to a 1D list
            elif not isinstance(value, (list, pd.Series, np.ndarray)):
                value = [value]

            # Convert the value to a DataFrame
            df = pd.DataFrame({key: value})

            # Construct the file path
            file_path = f'C:\\Users\\yfz5556\\PycharmProjects\\PromptEHR\\simulation_longICD\\{key}.csv'

            # Save the DataFrame to a CSV file
            df.to_csv(file_path, index=False)
    return
if __name__ == '__main__':

    input_dir = 'C:\\Users\\yfz5556\\PycharmProjects\\PromptEHR\\demo_data\\demo_patient_sequence\\ehr\\'
    # see(input_dir)
    # main(input_dir, True)
    # main(input_dir, True, False, mimic4=True)
    main(input_dir, True, False, False, True)