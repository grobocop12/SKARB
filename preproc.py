import pandas as pd
import matplotlib.pyplot as plt

'''data = pd.read_csv('genfile_file_seed145_winforcezyx_big_21_12_2018.csv', encoding= 'cp1250')'''
def preproc (data):
    data=data.dropna()
    '''data =data.drop(0)'''
    dated = data.where(abs(data['X'])<200000)
    dated = data.where(abs(dated['Z']) < 200000)
    dated = data.where(dated['X'] > -20000)
    print (data)

    print('post drop')
    print(dated)





    return dated
'''plt.figure(1)
preprocesed_data = preproc(data)
plt.scatter(preprocesed_data['X'],preprocesed_data['Z'])
plt.show()'''