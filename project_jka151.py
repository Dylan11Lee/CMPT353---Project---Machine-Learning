import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('CMPT353 Project jka151').getOrCreate()

assert sys.version_info >= (3, 4)
assert spark.version >= '2.1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


schema_wiki = types.StructType([
    types.StructField('imdb_id', types.StringType(), False),
    # types.StructField('rotten_tomatoes_id', types.StringType(), False),
    # types.StructField('wikidata_id', types.StringType(), False),
    types.StructField('label', types.StringType(), False),
    types.StructField('made_profit', types.BooleanType(), False),
    types.StructField('country_of_origin', types.StringType(), False),
    types.StructField('original_language', types.StringType(), False),
    # adapted from https://medium.com/@mrpowers/working-with-spark-arraytype-and-maptype-columns-4d85f3c8b2b3
    types.StructField('genre', types.ArrayType(types.StringType(), False), False),
    types.StructField('cast_member', types.ArrayType(types.StringType(), False), False),
    types.StructField('main_subject', types.ArrayType(types.StringType(), False), False),
    # types.StructField('director', types.ArrayType(types.StringType(), False), False),
    # types.StructField('filming_location', types.ArrayType(types.StringType(), False), False),
])

schema_rotten = types.StructType([
    types.StructField('imdb_id', types.StringType(), False),
    # types.StructField('rotten_tomatoes_id', types.StringType(), False),
    types.StructField('audience_average', types.StringType(), False),
    types.StructField('audience_percent', types.StringType(), False),
    types.StructField('audience_ratings', types.StringType(), False),
    types.StructField('critic_average', types.StringType(), False),
    types.StructField('critic_percent', types.StringType(), False),
])


def filter_null(spark_df):
    # adapted from https://stackoverflow.com/questions/35477472/difference-between-na-drop-and-filtercol-isnotnull-apache-spark
    spark_df = spark_df.filter(spark_df['label'].isNotNull())
    spark_df = spark_df.filter(spark_df['made_profit'].isNotNull())
    spark_df = spark_df.filter(spark_df['country_of_origin'].isNotNull())
    spark_df = spark_df.filter(spark_df['original_language'].isNotNull())
    spark_df = spark_df.filter(spark_df['genre'].isNotNull())
    spark_df = spark_df.filter(spark_df['cast_member'].isNotNull())
    spark_df = spark_df.filter(spark_df['main_subject'].isNotNull())
    
    spark_df = spark_df.filter(spark_df['audience_average'].isNotNull())
    spark_df = spark_df.filter(spark_df['audience_percent'].isNotNull())
    spark_df = spark_df.filter(spark_df['audience_ratings'].isNotNull())
    spark_df = spark_df.filter(spark_df['audience_ratings'] > 0)
    spark_df = spark_df.filter(spark_df['critic_average'].isNotNull())
    spark_df = spark_df.filter(spark_df['critic_percent'].isNotNull())
    return spark_df


def string_to_label(x):
    # adapted from http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
    le = preprocessing.LabelEncoder()
    le.fit(x)
    return le.transform(x)


def data_cleaning(df):
    # convert boolean value into int. 0 = false, 1 = true.
    # adapted from https://stackoverflow.com/questions/29960733/how-to-convert-true-false-values-in-dataframe-as-1-for-true-and-0-for-false
    df['made_profit'] = df['made_profit'].astype(int)
    # convert strings to ingegers
    df['label'] = string_to_label(df['label'])
    df['country_of_origin'] = string_to_label(df['country_of_origin'])
    df['original_language'] = string_to_label(df['original_language'])
    
    df['audience_average'] = df['audience_average'].astype(float)
    df['audience_percent'] = df['audience_percent'].astype(float)
    df['audience_ratings'] = df['audience_ratings'].astype(float)
    df['critic_average'] = df['critic_average'].astype(float)
    df['critic_percent'] = df['critic_percent'].astype(float)
    return df


def list_to_columns(df, col_name):
    # Creating new columns for each data(genre) and give 1 if present, 0 if not.
    # adapted from https://stackoverflow.com/questions/45312377/how-to-one-hot-encode-from-a-pandas-column-containing-a-list
    mlb = MultiLabelBinarizer()
    df = df.join(pd.DataFrame(mlb.fit_transform(df.pop(col_name)),
                          columns=mlb.classes_,
                          index=df.index))
    return df


def get_svr(df):
    X_columns = list(df)
    X_columns.remove('imdb_id')
    X_columns.remove('label')
    X_columns.remove('audience_average')
    
    y_column =  'audience_average'

    X = df[X_columns].values
    y = df[y_column].values
        
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # Support vector regression model
    model = make_pipeline(
        StandardScaler(),
        # https://stackoverflow.com/questions/41925157/logisticregression-unknown-label-type-continuous-using-sklearn-in-python
        SVR(kernel='rbf', C=1, epsilon=0.01)
    ) 
    model.fit(X_train, y_train)    
    return model.score(X_test, y_test)
    
def get_pca(df):
    X_columns = list(df)
    X_columns.remove('imdb_id')
    X_columns.remove('audience_average')
    X = df[X_columns].values
    
    flatten_model = make_pipeline(
        MinMaxScaler(),
        PCA(2)
    )
    X2 = flatten_model.fit_transform(X)
    assert X2.shape == (X.shape[0], 2)
    return X2

def get_clusters(df):
    X_columns = list(df)
    X_columns.remove('imdb_id')
    X_columns.remove('audience_average')
    X = df[X_columns].values
    
    model = make_pipeline(
        KMeans(n_clusters=10)
    )
    model.fit(X)
    return model.predict(X)


def plot_graph(df):
    X2 = get_pca(df)
    clusters = get_clusters(df)
    plt.scatter(X2[:, 0], X2[:, 1], c=clusters, cmap='Set1', edgecolor='k', s=20)
    plt.xlabel('X_PCA[:, 1]')
    plt.ylabel('X_PCA[:, 0]')
    

def main():
    # read json files
    data_wiki = spark.read.json(sys.argv[1], schema=schema_wiki)
    data_wiki = data_wiki.cache()
    data_rotten = spark.read.json(sys.argv[2], schema=schema_rotten)
    data_rotten = data_rotten.cache()
        
    # join two spark dataframes
    wiki_rotten = data_wiki.join(data_rotten, on='imdb_id')
    # filter out rows with specific nulls
    wiki_rotten = filter_null(wiki_rotten)
       
    # convert to pandas dataframe
    pd_data = wiki_rotten.toPandas() 
    # binarize boolean and string values
    pd_cleaned = data_cleaning(pd_data)
    
    pd_genre = pd_cleaned.drop(['cast_member', 'main_subject'], axis=1)    
    pd_genre = list_to_columns(pd_genre, 'genre')    
    
    pd_actor = pd_cleaned.drop(['genre', 'main_subject'], axis=1)    
    pd_actor = list_to_columns(pd_actor, 'cast_member')
    
    pd_subject = pd_cleaned.drop(['genre', 'cast_member'], axis=1)   
    pd_subject = list_to_columns(pd_subject, 'main_subject')      
    
    final_score = pd.DataFrame(
        {
            'Data': ['With Genres', 'With Actors', 'With Subjects'],
            'Score': [get_svr(pd_genre), get_svr(pd_actor), get_svr(pd_subject)]
        }
    )
    
    final_score.to_csv('final_score.csv', index=False)

    plot_graph(pd_genre)
    plt.title('PCA with Genres')
    plt.savefig('cluster_genre.png')
    plt.close()
    
    plot_graph(pd_actor)
    plt.title('PCA with Actors')
    plt.savefig('cluster_actor.png')
    plt.close()

    plot_graph(pd_subject)
    plt.title('PCA with Subjects')
    plt.savefig('cluster_subject.png')
    plt.close()


if __name__ == '__main__':
    main()

