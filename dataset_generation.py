import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime

# Functions
def injurie_age(season, birth):
    return datetime.datetime.strptime(season, '%Y-%m-%d').year - datetime.datetime.strptime(birth, '%Y-%m-%d').year

def define_bmi(height, weight):
    return 1 if (weight) / (height * height) > 28 else 0

def club_to_category(value):
    if 0 < value <= 75:
        return 1
    elif 75 < value <= 125:
        return 2
    elif 125 < value <= 225:
        return 3
    elif 225 < value <= 275:
        return 4
    elif 275 < value <= 600:
        return 5
    else:
        return 6

# Load data
tuples = []
i = 0
with open('dataset/Final-player.txt') as f:
    for t in f:
        i += 1
        tuples.append(eval(t[1:-1]) if i == 2580 else eval(t[1:-2]))

df = pd.DataFrame(tuples)
df.columns = ["id", "name", "club", "club_value", "birth", "weight", "height", "country", "role", "foot", "transfers", "injuries"]

# Data cleaning
df = df[df['birth'].notnull()]
number_of_players = df.shape[0]
print(f"{number_of_players} players loaded.")

# Extract injuries
injuries_list = [injurie for injurie in df['injuries'].sum()]
s = pd.Series(injuries_list).value_counts()
print("Injury types with more than 75 occurrences:")
for injur in s[s > 75].index:
    print(injur)

# Plot pie chart for top 20 injuries
s[:20].plot(kind='pie', autopct='%1.1f%%', startangle=90, title='Top 20 Injuries')
plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
plt.show()

# Expand injuries into separate rows
s = df.apply(lambda x: pd.Series(x.iloc[11]), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'injurie'
df = df.drop(['injuries'], axis=1).join(s).reset_index(drop=True)
df = df[df['injurie'].notnull()]

# Split injury data
df_injurie = pd.DataFrame(df['injurie'].tolist(), index=df.index)
df = df.drop('injurie', axis=1).join(df_injurie)
df = df.rename(columns={0: "season", 1: "type", 2: "days"})
df.season = df.season.apply(lambda x: '20' + x[3:] + '-01-01')

# Calculate age
df['age'] = df.apply(lambda x: injurie_age(x['season'], x['birth']), axis=1)

# Age distribution
df[(df['age'] > 19) & (df['age'] < 36)]['age'].value_counts(normalize=True).sort_index().plot(kind='bar')
plt.title('Age Distribution of Injured Players (20-35)')
plt.show()

# Filter muscle injuries
muscle_injurie = [
    'Hamstring Injury', 'Muscular problems', 'Muscle Injury', 'Torn Muscle Fibre',
    'Adductor problems', 'Thigh Muscle Strain', 'Groin Injury', 'Muscle Fatigue',
    'Achilles tendon problems', 'Torn muscle bundle', 'Biceps femoris muscle injury'
]

df[(df['type'].isin(muscle_injurie)) & (df['age'] > 19) & (df['age'] < 36)]['age'].value_counts(normalize=True).sort_index().plot(kind='bar')
plt.title('Muscle Injuries Age Distribution (20-35)')
plt.show()

# Plot general vs muscle injuries
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.set_title('General Injuries vs Muscle Injuries')
width = 0.4

df[(df['age'] > 19) & (df['age'] < 36)]['age'].value_counts(normalize=True).sort_index().plot(kind='bar', color='red', ax=ax, width=width, position=1)
df[(df['type'].isin(muscle_injurie)) & (df['age'] > 19) & (df['age'] < 36)]['age'].value_counts(normalize=True).sort_index().plot(kind='bar', color='blue', ax=ax2, width=width, position=0)

ax.set_ylabel('General Injuries')
ax2.set_ylabel('Muscle Injuries')
plt.show()

# Clean 'days' column
df['days'] = df['days'].replace('?', np.nan)
df['days'] = pd.to_numeric(df['days'], errors='coerce')
df.dropna(subset=['days'], inplace=True)

# Plot recovery time by age for specific injuries
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
fig.suptitle('Injury Recovery Time According to Age', fontsize=20)
count = 0
for injury in ['Ankle Injury', 'Hamstring Injury', 'Cruciate Ligament Rupture']:
    df_grouped = df.groupby('type').get_group(injury)
    df_grouped.groupby('age')['days'].mean().plot(ax=axes.flat[count], title=injury, color='blue')
    count +=1 

df.groupby('age')['days'].mean().plot(ax=axes.flat[count], title="All Injuries", color='blue')
plt.show()

# Calculate BMI
df['height'] = df['height'].str.replace(",", ".").str.replace('\\xa0', "").astype(float)
df = df[df['weight'] != '-']
df['weight'] = df['weight'].astype(float)

df['bmi'] = df.apply(lambda x: define_bmi(x['height'], x['weight']), axis=1)
players_high_bmi = set(df[df['bmi'] == 1].groupby('name').groups.keys())

# Clean and categorize club values
df['club_value'] = df['club_value'].apply(lambda x: x.split(",")[0]).astype(int)
df['club_value'] = df['club_value'].apply(lambda x: 1000 if x == 1 else x).astype(int)

# Plot club value histogram
df['club_value'].plot(kind='hist', bins=25)
plt.title('Distribution of Club Values')
plt.show()

# Plot cumulative histogram
fig, ax = plt.subplots()
df['club_value'].plot(kind='hist', bins=40, cumulative=True, ax=ax)
plt.axhline(y=2000, color='r', linestyle='-')
plt.axhline(y=4000, color='r', linestyle='-')
plt.axhline(y=6000, color='r', linestyle='-')
plt.axhline(y=8000, color='r', linestyle='-')
plt.axhline(y=10000, color='r', linestyle='-')
plt.title('Cumulative Distribution of Club Values')
plt.show()

# Categorize club values
df['club_value'] = df['club_value'].apply(club_to_category)

# Plot recovery time by club budget
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
fig.suptitle('Injury Recovery Time According to Club Budget', fontsize=20)
# Filter muscle injuries
muscle_injurie = [
    'Hamstring Injury', 'Muscular problems', 'Muscle Injury', 'Torn Muscle Fibre',
    'Adductor problems', 'Thigh Muscle Strain', 'Groin Injury', 'Muscle Fatigue',
    'Achilles tendon problems', 'Torn muscle bundle', 'Biceps femoris muscle injury'
]
count = 0
for injury in muscle_injurie[:6]:
    try:
        # Group by injury type and calculate mean recovery days by club value
        df_grouped = df.groupby('type').get_group(injury)
        df_grouped.groupby('club_value')['days'].mean().plot(kind='bar', ax=axes.flatten()[count], title=injury)
        count += 1
    except KeyError:
        print(f"No data for injury type: {injury}")
    except ValueError as ve:
        print(f"Error processing injury type '{injury}': {ve}")

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# Remove duplicates based on 'name' column and keep the last occurrence
# df_cleaned = df.drop_duplicates(subset=['id'], keep='last')


# Save DataFrame to CSV
csv_path = 'dataset/player_injury_data.csv'
df.to_csv(csv_path, index=False)

print(f"CSV file created and saved at: {csv_path}")