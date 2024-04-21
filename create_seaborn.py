import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_clip_scores(directory):
    clip_scores = []
    for filename in os.listdir(directory):
        if filename.endswith(".pth"):
            filepath = os.path.join(directory, filename)
            scores = torch.load(filepath)
            clip_scores.append(scores)
    return clip_scores

directory_va = 'C:\\Users\\ferdawsi\\Documents\\EECS_542\\va_reproduction\\stage_2'
directory_mcmc = 'C:\\Users\\ferdawsi\\Documents\\EECS_542\\mcmc\\stage_2'


clip_scores_va = load_clip_scores(directory_va)
clip_scores_mcmc = load_clip_scores(directory_mcmc)


unflipped_scores_va = []
flipped_scores_va = []
for scores in clip_scores_va:
    unflipped_scores_va.extend(scores[0].tolist())
    flipped_scores_va.extend(scores[1].tolist())

unflipped_scores_mcmc = []
flipped_scores_mcmc = []
for scores in clip_scores_mcmc:
    unflipped_scores_mcmc.extend(scores[0].tolist())
    flipped_scores_mcmc.extend(scores[1].tolist())


sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))


sns.kdeplot(x=unflipped_scores_va, y=flipped_scores_va, cmap="Reds", shade=True, label="VA", alpha=0.9)


sns.kdeplot(x=unflipped_scores_mcmc, y=flipped_scores_mcmc, cmap="Blues", shade=True, label="mcmc", alpha=0.7)


max_score_va = max(max(unflipped_scores_va), max(flipped_scores_va))
max_score_mcmc = max(max(unflipped_scores_mcmc), max(flipped_scores_mcmc))
max_score = max(max_score_va, max_score_mcmc)
plt.plot([0, max_score], [0, max_score], linestyle="--", color="black")  

plt.xlabel("Unflipped")
plt.ylabel("Flipped")
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Visual Anagrams', markerfacecolor='red', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Ours', markerfacecolor='blue', markersize=10)]
plt.legend(handles=legend_elements, loc='lower right')

plt.show()

