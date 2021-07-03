
from processes import CGMYProcess, VGProcess
from contracts import GMWB
import pandas as pd
import numpy as np
import time

res = pd.DataFrame(np.zeros((6, 2)), columns=['VG', 'CGMY'])

vg = VGProcess()
cgmy = CGMYProcess()

contract_vg = GMWB(process=vg, spot=0.05)
contract_vg.set_integration_limits(left=-2.5, right=0.5)
contract_cgmy = GMWB(process=cgmy, spot=0.05)

penalty = np.arange(0, 0.06, 0.01)

for k in np.arange(0, 6):
    try:
        contract_vg.set_penalty(penalty=penalty[k])
        start_time = time.time()
        print('VG process with penalty = %0.3f\n' % penalty[k])
        res.iloc[k, 0] = contract_vg.fair_fee(method="mixed")
        stop_time = time.time()
        print('Took %s seconds to calculate. Fair fee is %f. Price is %f\n'
              % (stop_time - start_time, res.iloc[k, 0], contract_vg._price))

    except ValueError:
        res.iloc[k, 0] = -1

    try:
        contract_cgmy.set_penalty(penalty=penalty[k])
        start_time = time.time()
        print('CGMY process with penalty = %0.3f\n' % penalty[k])
        res.iloc[k, 1] = contract_cgmy.fair_fee(method="mixed")
        stop_time = time.time()
        print('Took %s seconds to calculate. Fair fee is %f. Price is %f\n'
              % (stop_time - start_time, res.iloc[k, 1], contract_cgmy._price))
    except ValueError:
        res.iloc[k, 1] = -1

print('Writing to csv file ...\n')
res.to_csv("D:/Python/GMWB/mixed.csv", index=False)
print('Done!')
