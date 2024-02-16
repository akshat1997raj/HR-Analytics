# %%
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# %%
df = pd.read_csv("train_LZdllcl.csv")

# %%
df1 = df.copy()
df1 = df1[df1['length_of_service']<=15]
df1 = df1[df1['age']<55]

# %%
from sklearn.preprocessing import LabelEncoder

encoder1 = LabelEncoder()
encoder2 = LabelEncoder()
encoder3 = LabelEncoder()
encoder4 = LabelEncoder()
encoder5 = LabelEncoder()
df1['department'] = encoder1.fit_transform(df1['department'])
df1['region'] = encoder2.fit_transform(df1['region'])
df1['education'] = encoder3.fit_transform(df1['education'])
df1['gender'] = encoder4.fit_transform(df1['gender'])
df1['recruitment_channel'] = encoder5.fit_transform(df1['recruitment_channel'])

# %%
## removing null values
df1 = df1[df1['education']!=3]
df1 = df1[df1['previous_year_rating'].isnull()==False]

# %%
df1.drop(columns=['employee_id'], inplace=True)
df1.reset_index(drop = True, inplace= True)

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_x = pd.DataFrame(scaler.fit_transform(df1.drop(columns=['is_promoted'])), columns=df1.drop(columns=['is_promoted']).columns)
df_x['is_promoted'] = df1['is_promoted']

# %%
for column in df_x.columns:
    if pd.api.types.is_numeric_dtype(df_x[column]):
        print(column)
        print(df_x[column].skew())
        fig, axs = plt.subplots(2,1, figsize=(8,8))
        axs[0].hist(df_x[column], density = True)
        axs[0].set_title('Histogram')

        x_mean, x_std = np.mean(df_x[column]), np.std(df_x[column])
        x_min, x_max = plt.xlim(np.min(df_x[column]), np.max(df_x[column]))
        x = np.linspace(x_min, x_max, 100)
        p = norm.pdf(x, x_mean, x_std)
        axs[0].plot(x, p, 'k', linewidth = 2)

        axs[1].boxplot(df_x[column], vert=False, patch_artist=True)
        axs[1].set_title('boxplot')

        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## XGB

# %%
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# %%
classifier = XGBClassifier()

# %%
x= df_x.drop(columns=['is_promoted']).reset_index(drop=True)
y= df_x['is_promoted'].reset_index(drop=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# %%
from imblearn.combine import SMOTETomek
os = SMOTETomek(sampling_strategy=0.75)
x_train_s, y_train_s  = os.fit_resample(x_train,y_train)

# %%
# Finding out best parameters for xgb

from sklearn.model_selection import GridSearchCV

param_grid2 = {
    'n_estimators':[50,150,250],
    'max_depth':[None, 5, 10, 15]
}

grid_search2 = GridSearchCV(estimator=classifier, param_grid=param_grid2, cv = 4, verbose=2)
grid_search2.fit(x_train_s,y_train_s)

# %%
grid_search2.best_params_

# %%
classifier_gs_xgb = grid_search2.estimator
classifier_gs_xgb.fit(x_train_s, y_train_s)

# %%
y_pred =classifier_gs_xgb.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy

# %%
report = classification_report(y_test, y_pred)
print(report)

# %%
print(classification_report(y_train_s, classifier_gs_xgb.predict(x_train_s)))

# %%
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

# %%
df_sub = pd.read_csv('test_2umaH9m.csv')

# %%
print(encoder1.classes_)
print(encoder2.classes_)
print(encoder3.classes_)
print(encoder4.classes_)
print(encoder5.classes_)

# %%
df_sub

# %%
df_sub['department'] = encoder1.transform(df_sub['department'])
df_sub['region'] = encoder2.transform(df_sub['region'])
df_sub['education'] = encoder3.transform(df_sub['education'])
df_sub['gender'] = encoder4.transform(df_sub['gender'])
df_sub['recruitment_channel'] = encoder5.transform(df_sub['recruitment_channel'])

# %%
df_sub_std = df_sub.drop(columns=['employee_id'])
df_sub_std = pd.DataFrame(scaler.transform(df_sub_std), columns=df_sub_std.columns)
y_pred_sub = classifier.predict(df_sub_std)

# %%
df_sub['is_promoted'] =  y_pred_sub
df_sub_final = df_sub[['employee_id','is_promoted']]
df_sub_final.to_csv('sub3.csv', index=False)


