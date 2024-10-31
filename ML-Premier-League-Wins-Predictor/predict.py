import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create a DataFrame for easy plotting
importance_df = pd.DataFrame({
    'Feature': [
        'goals', 'head_clearance', 'clean_sheet', 'ontarget_scoring_att', 'total_red_card',
        'pen_goals_conceded', 'dispossessed', 'punches', 'att_freekick_goal', 'penalty_save',
        'saves', 'total_cross', 'total_long_balls', 'backward_pass', 'total_yel_card',
        'total_high_claim', 'clearance_off_line', 'big_chance_missed', 'total_through_ball',
        'last_man_tackle', 'goal_fastbreak', 'own_goals', 'hit_woodwork', 'total_pass',
        'total_offside', 'att_hd_goal', 'touches', 'interception', 'att_pen_goal',
        'total_tackle', 'corner_taken', 'outfielder_block', 'penalty_conceded',
        'total_scoring_att', 'losses', 'total_clearance', 'goals_conceded',
        'att_obox_goal', 'att_ibox_goal'
    ],
    'Coefficient': [
        18.199322, 1.173094, 0.964848, 0.882167, 0.342551,
        0.290625, 0.196494, 0.154717, 0.142457, 0.120781,
        0.094644, 0.078300, 0.073135, 0.067982, 0.057790,
        0.051065, 0.043670, 0.037300, 0.015940, -0.047705,
        -0.068439, -0.077506, -0.118678, -0.150763, -0.216695,
        -0.227477, -0.231309, -0.280548, -0.289247, -0.317119,
        -0.326504, -0.335374, -0.375929, -0.426683, -0.686558,
        -1.010426, -1.227266, -3.139490, -12.955950
    ]
})

# Sort the DataFrame by absolute values of coefficients for plotting
importance_df['abs_Coefficient'] = importance_df['Coefficient'].abs()
importance_df = importance_df.sort_values(by='abs_Coefficient', ascending=False)

# Plotting
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Coefficient'], color='skyblue')
plt.axvline(0, color='grey', lw=0.8)
plt.title('Feature Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
