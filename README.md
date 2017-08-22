
The dataset, and further information, can be found at

http://archive.ics.uci.edu/ml/datasets/mammographic+mass

This is a binary classification task; we would like to predict whether a mammographic mass is benign or malignant. 


| Name | Type | Predictive/Output | Description | 
| ---- | ----- | ------- | ------- |
| birads | ordinal | non-predictive |  assessment: 1 to 5 |
| age | integer | predictive | patient's age in years |
| shape | nominal | predictive | mass shape: round=1, oval=2, lobular=3, irregular=4 | 
| margin | nominal | predictive | mass margin: circumscribed=1, microlobulated=2, obscured=3, ill-defined=4, spiculated=5 |
| density | ordinal | predictive | mass density: high=1, iso=2, low=3, fat-containing=4 |
| Severity | binomial | output | benign=0, malignant=1 |

```````````
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
```````````

```````````
df = pd.read_csv('mammographic_masses.csv', sep=',', header=None)
df.shape
Out[8]: (961, 6)
newdf <- df.replace(to.replace=‘?’,value=‘ ‘)
```````````
Our next task is to determine the number of missing or mistyped values for each categorical predictive feature. 
`````````
pd.crosstab(index=newdf["birads"], columns="count")
pd.crosstab(index=newdf["shape"], columns="count")
pd.crosstab(index=newdf["margin"], columns="count")
pd.crosstab(index=newdf["density"], columns="count")
`````````

|birads|count|      
|------|-----|    
|      |  2 |       
|0   |        5|    
|2   |       14|
|3   |       36|
|4   |      547|
|5   |      345|
|55  |        1|
|6   |       11|

|shape|count|
|-----|-----|
|     |   31|
|1    |  224|
|2    |  211|
|3    |  95 |
|4    | 400 |

|margin|count|    
|------|-----|
|      | 48  |
|1     | 357|
|2      |    24|
|3      |   116|
|4      |   280|
|5      |   136|

|density|count|       
|-------|-----|
|       |  76 |
|1      |  16 |
|2      |  59 |
|3      | 798 |
|4      | 12  |

Since the corresponding table for the "age" feature is very long, we mention only that it has 5 missing values.  Based on this and the fact that the "birad" feature has only 2 missing values, we remove the corresponding instances as well as the instance where the birad entry is 55.
````````
df1 = newdf[newdf.birads != ' ']
df1 = df1[df1.birads != '55']
df1 = df1[df1.age != ‘ ‘] 
df1.shape
  (953,6)
````````
The number of missing values for shape, margin and density are fairly large.  We now determine the number of instances with missing values for at least one of "shape","margin" or "density."
````````
dfs = df1[df1["shape"] != ' ']
dfm = dfs[dfs["margin"] != ' ']
dfd = dfm[dfm["density"] != ' ']
dfd.shape
  (829, 6)
````````
It follows that 124 of the 953 instances in df1 have at least 1 missing value; this justifies filling in the missing values.  We will do this using probabilities that condition on "birad" and "age", neither of which have missing values in df1.  First we produce a histogram for "age."

          



