import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path_prostate = r'../results/prostate/prostate_results.csv'
file_path_bladder = r'../results/bladder/bladder_results.csv'
file_path_rectum = r'../results/rectum/rectum_results.csv'
file_path_multiclass = r'../results/multiclass/multiclass_results.csv'

df_prostate = pd.read_csv(file_path_prostate)
df_bladder = pd.read_csv(file_path_bladder)
df_rectum = pd.read_csv(file_path_rectum)
df_multiclass = pd.read_csv(file_path_multiclass)

# matplotlib
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
# prostate
vp1_prostate = ax1.violinplot(df_prostate['dice'])
vp2_prostate = ax1.violinplot(df_prostate['iou'])
# bladder
vp1_bladder = ax2.violinplot(df_bladder['dice'])
vp2_bladder = ax2.violinplot(df_bladder['iou'])
# rectum
vp1_rectum = ax3.violinplot(df_rectum['dice'])
vp2_rectum = ax3.violinplot(df_rectum['iou'])
# multiclass
vp1_multiclass = ax4.violinplot(df_multiclass['dice'])
vp2_multiclass = ax4.violinplot(df_multiclass['iou'])
# Add a legend
ax1.legend([vp1_prostate['bodies'][0], vp2_prostate['bodies'][0]], ['DSC', 'JI'], loc=2)
# Set title names
ax1.set_title('Prostate')
ax2.set_title('Bladder')
ax3.set_title('Rectum')
ax4.set_title('Multiclass')
# Remove y-axis
ax2.label_outer()
ax3.label_outer()
ax4.label_outer()
# Remove padding and show
plt.tight_layout()
plt.show()
