import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load datasets
train_df = pd.read_csv("datasets/train.csv")
test_df = pd.read_csv("datasets/test.csv")

# Mark train/test sets
train_df['TrainSet'] = True
test_df['TrainSet'] = False
test_df['Survived'] = None  

# Combine datasets
combined = pd.concat([train_df, test_df], sort=False)

# Handle missing values
combined['Embarked'] = combined['Embarked'].fillna(combined['Embarked'].mode()[0])
combined['Fare'] = combined['Fare'].fillna(combined['Fare'].median())
combined['Age'] = combined['Age'].fillna(combined['Age'].median())

# Drop irrelevant or missing columns
combined = combined.drop(columns=['Cabin', 'Ticket', 'Name'])

# Encode categorical variables
combined['Sex'] = combined['Sex'].map({'male': 0, 'female': 1})
combined = pd.get_dummies(combined, columns=['Embarked'], drop_first=True)

# Split cleaned data
train_cleaned = combined[combined['TrainSet'] == True].drop(columns=['TrainSet'])
test_cleaned = combined[combined['TrainSet'] == False].drop(columns=['TrainSet', 'Survived'])

# Create folder for plots
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

# --- PLOTS ---

sns.set(style="whitegrid")

# Plot 1: Survival Count
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=train_cleaned,hue='Survived', palette={0: '#FF0000', 1: '#00AA00'})
plt.title('Survival Count')
plt.xticks([0, 1], ['Died', 'Survived'])
plt.ylabel('Count')
plt.xlabel('Survival')
plt.savefig(f"{output_dir}/survival_count.png",bbox_inches='tight')
plt.close()

# Plot 2: Survival by Sex 
plt.figure(figsize=(6, 4))
sns.barplot(x='Sex', y='Survived', data=train_cleaned, hue='Sex', palette='pastel', legend=False)
plt.title('Survival Rate by Sex')
plt.xticks([0, 1], ['Male', 'Female'])
plt.ylabel('Survival Rate')
plt.xlabel('Sex')
plt.savefig(f"{output_dir}/survival_by_sex.png",bbox_inches='tight')
plt.close()

# Plot 3: Survival by Pclass
plt.figure(figsize=(6, 4))
sns.barplot(x='Pclass', y='Survived', data=train_cleaned, hue='Pclass', palette='pastel', legend=False)
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.xlabel('Pclass')
plt.savefig(f"{output_dir}/survival_by_pclass.png",bbox_inches='tight')
plt.close()

# Plot 4: Age Distribution
plt.figure(figsize=(8, 4))
sns.histplot(
    data=train_cleaned,
    x='Age',
    hue='Survived',
    bins=30,
    element='step',
    stat='count',
    hue_order=[0, 1],
    palette={0: '#FF0000', 1: '#00AA00'}
)
plt.xticks(ticks=range(0, 90, 10))
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['Died', 'Survived'])
plt.savefig(f"{output_dir}/age_distribution_step_count.png",bbox_inches='tight')
plt.close()

# Plot 5: Fare Distribution by Survival 
plt.figure(figsize=(8, 4))
sns.histplot(
    data=train_cleaned,
    x='Fare',
    hue='Survived',
    bins=40,
    element='step',
    stat='count',
    common_norm=False,
    hue_order=[0, 1],
    palette={0: '#FF0000', 1: '#00AA00'}
)
plt.title('Fare Distribution by Survival')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['Died', 'Survived'])
plt.xlim(0, 200)
plt.savefig(f"{output_dir}/fare_distribution_by_survival.png",bbox_inches='tight')
plt.close()



# survival to numeric since its object
train_cleaned['Survived'] = train_cleaned['Survived'].astype(int)

# Plot 6: Correlation Heatmap
plt.figure(figsize=(10, 6))
corr = train_cleaned.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig(f"{output_dir}/correlation_heatmap.png",bbox_inches='tight')
plt.close()


# Plot 7: Pairplot
selected = ['Survived', 'Age', 'Fare', 'Pclass', 'Sex', 'SibSp', 'Parch']
pairplot = sns.pairplot(train_cleaned[selected], hue='Survived', palette='pastel', corner=True)
pairplot.fig.suptitle("Pairplot of Titanic Features", y=1.02)
pairplot.savefig(f"{output_dir}/pairplot_selected.png",bbox_inches='tight')
plt.close()


# Create a folder for CSV outputs 
csv_output_dir = "outputs"
os.makedirs(csv_output_dir, exist_ok=True)

# Save cleaned training data to CSV
train_csv_path = os.path.join(csv_output_dir, "cleaned_train_data.csv")
train_cleaned.to_csv(train_csv_path, index=False)

# Save cleaned test data to CSV
test_csv_path = os.path.join(csv_output_dir, "cleaned_test_data.csv")
test_cleaned.to_csv(test_csv_path, index=False)
