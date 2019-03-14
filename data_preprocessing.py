import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if substring in big_string:
            return substring
    return np.nan

def processFile(fileName):
    df=pd.io.parsers.read_csv(fileName)
    return df

def binaryTransformSex(df):
    if df['Sex']=='Male' or df['Sex']=='male':
        return 0
    elif df['Sex']=='Female' or df['Sex']=='female':
        return 1

def replace_titles(x):
    title=x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title

def processData(df):
    title_list_map = {'Mrs':0, 'Mr':1, 'Master':2, 'Miss':3, 'Special':4, 'Rev':5,
                    'Dr':6, 'Ms':7, 'Mlle':8,'Col':9, 'Capt':10, 'Mme':11, 'Countess':12,
                    'Don':13, 'Jonkheer':14}
    title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']
    cabins_map = {'A':0,'B':0,'C':3,'D':4,'E':5,'F':6,'T':7,'G':8,'Unknown':9}
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    embarkMap = {'C':0,'Q':1,'S':2,'Unknown':3}
    df['Sex'] = df.apply(binaryTransformSex,axis=1)
    df['Title']=df['Name'].map(lambda x: substrings_in_string(x, title_list))
    #df['Title']=df.apply(replace_titles, axis=1)
    #df['Title'] =df['Title'].map(lambda x: title_list_map[substrings_in_string(x, title_list)])

    df['Title'] = df['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
    df['Title'] = df['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',
                                            'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')
    df['Cabin']=df['Cabin'].fillna('Unknown')
    df['Title'] =df['Title'].map(lambda x: title_list_map[x])
    df['Deck'] = df['Cabin'].map(lambda x: cabins_map[substrings_in_string(x, cabin_list)])
    df['Embarked']=df['Embarked'].fillna('S')
    df['Embarked'] = df['Embarked'].map(lambda x: embarkMap[x])
    #df['has_Cabin'] = ~df.Cabin.insnull()
    df['Age']=df.Age.fillna(df.Age.mean())
    df['Fare']=df.Fare.fillna(df.Fare.mean())
    df['Family_Size']=df['SibSp']+df['Parch']
    #df['Age*Class']=df['Age']*df['Pclass']
    #df['Age']=df['Age'].fillna(0)
    df['Fare_Per_Person']=df['Fare']/(df['Family_Size']+1)
    df=df.drop("Name",axis=1)
    df=df.drop("Cabin",axis=1)
    df=df.drop('Ticket',axis=1)
    df=df.drop('SibSp',axis=1)
    df=df.drop('Parch',axis=1)

    return df
