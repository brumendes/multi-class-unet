import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,5)

file_path = r'../results/Results.xlsx'
df = pd.read_excel(
    file_path, 
    sheet_name=['Prostate', 'Bladder', 'Rectum', 'Multiclass'],
    usecols=['time', 'acc', 'dice', 'iou', 'lr', 'train loss', 'val loss']
    )

organ = 'Multiclass'

plt.plot(df[organ].index, df[organ]['acc'], 'c', label='Accuracy')
plt.plot(df[organ].index, df[organ]['dice'], 'm', label='DSC')
plt.plot(df[organ].index, df[organ]['iou'], 'y', label='JI')
plt.plot(df[organ].index, df[organ]['train loss'], 'g', label='Training loss')
plt.plot(df[organ].index, df[organ]['val loss'], 'b', label='Validation loss')
plt.xlabel('Epochs')
plt.legend()
plt.tight_layout()
plt.show()

# plt.plot(df[organ].index, df[organ]['lr'], 'g')
# plt.xlabel('Epochs')
# plt.ylabel('Learning Rate')
# plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
# plt.tight_layout()
# plt.show()