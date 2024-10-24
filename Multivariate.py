import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_age_interval_by_pclass(df):
    """
    Plot count of Age Interval grouped by Pclass
    
    Args:
    df (pd.DataFrame): DataFrame containing 'Age Interval' and 'Pclass'
    """
    sns.countplot(x='Age Interval', hue='Pclass', data=df, palette='Set2')
    plt.title('Count of Age Interval grouped by Pclass')
    plt.grid(True, linestyle='-.', axis='y')
    plt.show()


def plot_age_interval_by_embarked(df):
    """
    Plot count of Age Interval grouped by Embarked
    
    Args:
    df (pd.DataFrame): DataFrame containing 'Age Interval' and 'Embarked'
    """
    sns.countplot(x='Age Interval', hue='Embarked', data=df, palette='Set3')
    plt.title('Count of Age Interval grouped by Embarked')
    plt.grid(True, linestyle='-.', axis='y')
    plt.show()


def plot_pclass_by_fare_interval(df):
    """
    Plot count of Pclass grouped by Fare Interval
    
    Args:
    df (pd.DataFrame): DataFrame containing 'Pclass' and 'Fare Interval'
    """
    sns.countplot(x='Pclass', hue='Fare Interval', data=df, palette='Set1')
    plt.title('Count of Pclass grouped by Fare Interval')
    plt.grid(True, linestyle='-.', axis='y')
    plt.show()


def set_family_type(df):
    """
    Create Family Type feature based on Family Size (Single, Small, Large)
    
    Args:
    df (pd.DataFrame): DataFrame containing 'Family Size'
    
    Returns:
    df (pd.DataFrame): DataFrame with 'Family Type' column added
    """
    df['Family Type'] = df['Family Size']
    df.loc[df['Family Size'] == 1, 'Family Type'] = 'Single'
    df.loc[(df['Family Size'] > 1) & (df['Family Size'] < 5), 'Family Type'] = 'Small'
    df.loc[df['Family Size'] >= 5, 'Family Type'] = 'Large'
    return df


def set_titles(df):
    """
    Standardize Titles by unifying common titles (Mr, Mrs, Miss, Rare)
    
    Args:
    df (pd.DataFrame): DataFrame containing 'Title'
    
    Returns:
    df (pd.DataFrame): DataFrame with standardized 'Titles'
    """
    df['Titles'] = df['Title']
    df['Titles'] = df['Titles'].replace(['Mlle.', 'Ms.'], 'Miss.')
    df['Titles'] = df['Titles'].replace('Mme.', 'Mrs.')
    df['Titles'] = df['Titles'].replace(
        ['Lady.', 'the Countess.', 'Capt.', 'Col.', 'Don.', 'Dr.', 
         'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'Dona.'], 'Rare'
    )
    return df


def group_titles_by_sex_and_survival(df):
    """
    Group by Titles, Sex and Survived, and calculate survival mean
    
    Args:
    df (pd.DataFrame): DataFrame containing 'Titles', 'Sex', 'Survived'
    
    Returns:
    pd.DataFrame: Grouped DataFrame with mean of survival rate
    """
    return df[['Titles', 'Sex', 'Survived']].groupby(['Titles', 'Sex'], as_index=False).mean()