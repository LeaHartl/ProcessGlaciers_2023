import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

errors1997 = pd.read_csv('outputXdem/19691997_error_dataframe.csv')
errors2006 = pd.read_csv('outputXdem/19972006_error_dataframe.csv')
errors2017 = pd.read_csv('outputXdem/20062017_error_dataframe.csv')

errors1997.loc[errors1997.nrsamples>100] = np.nan

combined_errors = pd.DataFrame({'1996-97': errors1997.dh_err,
                                '1997-06': errors2006.dh_err,
                                '2006-17': errors2017.dh_err})

combined_neff = pd.DataFrame({'neff97': errors1997.nrsamples,
                                'neff06': errors2006.nrsamples,
                                'neff17': errors2017.nrsamples})

print(combined_errors.median())
print(combined_errors.max())

fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharey=True)

ax.scatter(errors1997.slope, errors1997.dh_err, label='1969-97', color='darkgrey', s=8, marker='s')
ax.scatter(errors2006.slope, errors2006.dh_err, label='1997-2006', color='k', s=8, marker='o')
ax.scatter(errors2017.slope, errors2017.dh_err, label='2006-17', color='slategray', s=8, marker='^')
ax.grid('both')
ax.set_xlabel('Mean slope (°) per glacier')
ax.set_ylabel('Uncertainty of mean elevation change (m) per glacier')

ax.legend()
fig.savefig('plots/errors_overview.png', bbox_inches='tight', dpi=300)
plt.show()

