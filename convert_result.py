import pandas as pd
import ast

dir_path = './simulation_shortICD/'
visit_file = dir_path + 'visit.csv'
label_file = dir_path + 'y.csv'

visits = pd.read_csv(visit_file)
labels = pd.read_csv(label_file)
res = pd.DataFrame()

def convert_to_4_modalities(row):
        row = ast.literal_eval(row)
        diag, drug, lab, proc, timegap = [], [], [], [], []
        time = 0
        for visit in row:
            diag.append(visit[0])
            drug.append(visit[1])
            lab.append(visit[2])
            proc.append(visit[3])
            timegap.append(time)
            time += 1
        return diag, drug, lab, proc, timegap

res = visits['visit'].apply(convert_to_4_modalities).apply(pd.Series)

# Now you have a DataFrame 'res' where each row has a tuple with four elements.
# We need to assign these elements to separate columns.
res.columns = ['DIAGNOSES_int', 'DRG_CODE_int', 'LAB_ITEM_int', 'PROC_ITEM_int', 'time_gaps']

# Concatenate the labels DataFrame to the transformed visits DataFrame
combined_data = pd.concat([res, labels.reset_index(drop=True)], axis=1)
#rename y to MORTALITY
combined_data = combined_data.rename(columns={'y': 'MORTALITY'})

# Export the combined DataFrame to a new CSV file
combined_data.to_csv(dir_path + 'promptEHR_synthetic_3digmimic.csv', index=False)

print("Combined data saved")