import h5py
import json
import os

models = ['static/models/pipe2.hdf5','static/models/pipe3.hdf5','static/models/pipe4.hdf5']

for m in models:
    print('\n===', m, '===')
    if not os.path.exists(m):
        print('MISSING')
        continue
    try:
        with h5py.File(m, 'r') as f:
            print('Keys:', list(f.keys()))
            # attributes
            for k,v in f.attrs.items():
                try:
                    print('attr', k, ':', v)
                except Exception as e:
                    print('attr', k, ': (non-printable)')
            if 'model_config' in f:
                try:
                    mc = f['model_config'][()]
                    if isinstance(mc, bytes):
                        mc = mc.decode('utf-8')
                    print('model_config (first 500 chars):')
                    print(mc[:500])
                except Exception as e:
                    print('Could not read model_config:', e)
            else:
                print('No model_config dataset; maybe saved weights-only or custom structure')
    except Exception as e:
        print('Error opening file:', e)
