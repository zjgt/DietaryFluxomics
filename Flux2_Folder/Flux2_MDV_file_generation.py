import pandas as pd

df1 = pd.read_csv('Flux2_mouse_Pcx_fractional_abundance_for_MDV.csv', header=0)
df3 = pd.read_csv('Flux2_MDV_template_mouse_Pcx.csv', header=0)

l1, w1 = df1.shape
l3, w3 = df3.shape
list1 = list(df1)
del list1[0:3]

dest_dir1 = 'Flux2_MDV_Mouse_Pcx_Files/'

for lists in list1:
    file_path = dest_dir1 + lists
    df3['MDV'] = df1[lists].values
    df3.to_csv(file_path, sep='\t', index=False)






