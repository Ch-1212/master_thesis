import pandas as pd


ari = pd.read_csv(r'data/parameter_results/sax_ari_parameter_results.csv')
ami = pd.read_csv(r'data/parameter_results/sax_ami_parameter_results.csv')
fow_mal = pd.read_csv(r'data/parameter_results/sax_fow_mal_parameter_results.csv')

#highest values
# ami: alph = 3; paa = 4, 7, 10, 12, 19, 22
# ari: alph = 3; paa = 4, 7, 10, 12, 19, 22
# fow_mal: alph = 3; paa = 4, 7, 10, 12, 19, 22
ari.index = ari['Unnamed: 0']
ami.index = ami['Unnamed: 0']
fow_mal.index = fow_mal['Unnamed: 0']

ari = ari.drop(['Unnamed: 0'], axis=1)
ami = ami.drop(['Unnamed: 0'], axis=1)
fow_mal = fow_mal.drop(['Unnamed: 0'], axis=1)

ari = ari.round(4)
ami = ami.round(4)
fow_mal = fow_mal.round(4)

ari.to_csv(r'data/parameter_results/sax_ari_rounded_parameter_results.csv')
ami.to_csv(r'data/parameter_results/sax_ami_rounded_parameter_results.csv')
fow_mal.to_csv(r'data/parameter_results/sax_fow_mal_rounded_parameter_results.csv')

ari_max = ari.max()
s = ari.max()[ari.max() == ari.max(1).max()].index
s = str(s[0])
max_index = ari.idxmax()[s]

print("Finished parameter test")