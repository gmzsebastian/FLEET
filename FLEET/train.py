from FLEET.classify import predict_SLSN
from multiprocessing import Pool
from astropy import table
import numpy as np
import glob
import os

all_objects = table.Table.read('sne_best.txt', format = 'ascii')

def predict_individual(i, date_range = 70, model = 'double', overwrite_features = False):
    print('                                                                                                                    ', i, '/', len(all_objects))
    name = all_objects[i]['Name']
    ra   = float(all_objects[i]['RA'])
    dec  = float(all_objects[i]['DEC'])
    z    = float(all_objects[i]['Redshift'])

    output = predict_SLSN(name, ra, dec, redshift = z, plot_lightcurve = True, date_range = date_range, model = model, import_lightcurve = False, 
                          save_features = True, overwrite_features = overwrite_features)

def join_table(date_range, model):
    tables = glob.glob('%s_%s/*.txt'%(date_range, model))

    final_table = table.Table()
    for item in tables:
        print(item)
        data = table.Table.read(item, format = 'ascii')
        data = table.Table(data, dtype = ['str'] * len(data[0]))
        final_table = table.vstack([final_table, data])

    classes_index = [np.where(all_objects['Name'] == i)[0][0] for i in final_table['object_name']]
    classes = table.Column(all_objects[classes_index]['Classification'], name = 'class')

    final_table.add_column(classes)
    final_table.write('training_set/center_table_%s_%s.txt'%(date_range, model), format = 'ascii')

if len(glob.glob('training_set')) == 0:
    os.system('mkdir training_set')

pool = Pool(25)
pool.map(predict_individual, np.arange(len(all_objects)))
join_table(70, 'double')
#join_table(20, 'single')
