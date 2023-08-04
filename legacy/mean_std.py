import os
import json
import pandas as pd

# Create an empty DataFrame to store the scores
df = pd.DataFrame()
root_predictions_dir = '/local_storage/users/paulbp/xview2/predictions/'
# Loop over each submission directory
submission_dirs = list([f'no_tune_no1_seed{i}'for i in (0, 1, 23)]
                       + [f'retrain34_imagenet_seed{i}'for i in (0, 1, 23)]
                       + [f'train34_no1_seed{i}' for i in (0, 1, 23)]
                       + [f'train34_integrated_seed{i}' for i in (0, 1, 23)]
                       + [f'integrated_distr_equal_lowaug_nodil_seed{i}' for i in (0, 1, 23)]
                       + [f'integrated_distr_equal_lowaug_seed{i}' for i in (0, 1, 23)]
                       + [f'integrated_distr_equal_seed{i}' for i in (0, 1, 23)])
for submission_dir in submission_dirs:
    print(f'Processing {submission_dir}...')
    submission_path = os.path.join(root_predictions_dir, submission_dir,
                                   f'submission_resnet34_{submission_dir}_{submission_dir.split("seed")[1]}')

    # Load the scores from the metrics.json file into a dictionary
    metrics_path = os.path.join(submission_path, 'metrics.json')
    with open(metrics_path, 'r') as f:
        scores_dict = json.load(f)

    # Convert the dictionary to a DataFrame
    scores_df = pd.DataFrame(scores_dict, index=[0])

    # Add a column for the submission directory
    scores_df['submission_dir'] = submission_dir

    # Append the scores to the main DataFrame
    df = df.append(scores_df)


# compute the mean and std of group of scores group are defined by the 
# submission_dir.split('seed')[0] for all the  columns (
# score,damage_f1,localization_f1,damage_f1_no_damage,
# damage_f1_minor_damage,damage_f1_major_damage,damage_f1_destroyed)

df = df.groupby(df.submission_dir.str.split(
    '_seed').str[0]).agg(['mean', 'std'])

# put std and mean in a same column where you have "mean ± std" 
# with 3 decimals for the mean and 2 for the std

df.columns = ['_'.join(col).strip() for col in df.columns.values]
df = df.round({'damage_f1_mean': 3, 'damage_f1_std': 2,
               'localization_f1_mean': 3, 'localization_f1_std': 2,
               'damage_f1_no_damage_mean': 3, 'damage_f1_no_damage_std': 2,
               'damage_f1_minor_damage_mean': 3, 'damage_f1_minor_damage_std': 2,
               'damage_f1_major_damage_mean': 3, 'damage_f1_major_damage_std': 2,
               'damage_f1_destroyed_mean': 3, 'damage_f1_destroyed_std': 2,
               'score_mean': 3, 'score_std': 2})
df['damage_f1_mean'] = df['damage_f1_mean'].astype(
    str) + ' ± ' + df['damage_f1_std'].astype(str)
df['localization_f1_mean'] = df['localization_f1_mean'].astype(
    str) + ' ± ' + df['localization_f1_std'].astype(str)
df['damage_f1_no_damage_mean'] = df['damage_f1_no_damage_mean'].astype(
    str) + ' ± ' + df['damage_f1_no_damage_std'].astype(str)
df['damage_f1_minor_damage_mean'] = df['damage_f1_minor_damage_mean'].astype(
    str) + ' ± ' + df['damage_f1_minor_damage_std'].astype(str)
df['damage_f1_major_damage_mean'] = df['damage_f1_major_damage_mean'].astype(
    str) + ' ± ' + df['damage_f1_major_damage_std'].astype(str)
df['damage_f1_destroyed_mean'] = df['damage_f1_destroyed_mean'].astype(
    str) + ' ± ' + df['damage_f1_destroyed_std'].astype(str)
df['score_mean'] = df['score_mean'].astype(
    str) + ' ± ' + df['score_std'].astype(str)
df = df.drop(['damage_f1_std', 'localization_f1_std', 'damage_f1_no_damage_std', 'damage_f1_minor_damage_std',
              'damage_f1_major_damage_std', 'damage_f1_destroyed_std', 'score_std'], axis=1)


print(df)
# Save the scores to a csv file
df.to_csv('scores.csv')
