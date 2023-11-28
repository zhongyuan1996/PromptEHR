import os
import dill
import pickle
import pandas as pd
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
input_dir = 'C:\\Users\\zhong\\PycharmProjects\\PromptEHR\\demo_data\\demo_patient_sequence\\ehr\\'

# see the input format
# demo = load_mimic_ehr_sequence(input_dir=input_dir, n_sample=100)

def loadingData(name):
    data = pd.read_csv(input_dir + name + '_mimic.csv')
    visit = data['visit'].apply(lambda x: ast.literal_eval(x)).tolist()
    for i in range(len(visit)):
        for j in range(len(visit[i])):
            visit[i][j] = [visit[i][j]]
    voc = pd.read_csv(input_dir + 'code_to_int_mapping.csv')
    feature = data.drop(columns='visit')
    label = feature['MORTALITY'].values
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
            'order':['diag'],
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

# print('visit', demo['visit'][0]) # a list of visit events
# print('mortality', demo['mortality'][0]) # array of labels
# print('feature', demo['feature'][0]) # array of patient baseline features
# print('voc', demo['voc']) # dict of dicts containing the mapping from index to the original event names
# print('order', demo['order']) # a list of three types of code
# print('n_num_feature', demo['n_num_feature']) # int: a number of patient's numerical features
# print('cat_cardinalities', demo['cat_cardinalities']) # list: a list of cardinalities of patient's categorical features

train_seq, train_demo = loadingData('train')
# toy_seq, toy_demo = loadingData('toy')
test_seq, test_demo = loadingData('test')
val_seq, val_demo = loadingData('val')

# fit the model
model = PromptEHR(
    code_type=test_demo['order'],
    n_num_feature=test_demo['n_num_feature'],
    cat_cardinalities=test_demo['cat_cardinalities'],
    num_worker=0,
    eval_step=1,
    epoch=1,
    device=[1,2],
)
model.fit(
    train_data=train_seq,
    val_data=test_seq,
)