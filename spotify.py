
from getpass import getpass
import os
from dotenv import load_dotenv
load_dotenv()


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

client_credentials_manager = SpotifyClientCredentials(client_id = CLIENT_ID, client_secret = CLIENT_SECRET)

sp = spotipy.Spotify(client_credentials_manager= client_credentials_manager)


def get_playlist_data(my_playlist_link):
    my_playlist_link = input("Enter playlist URL:")
    my_playlist_URI = my_playlist_link.split("/")[-1].split("?")[0]

    tracks_in_my_playlist_info = sp.playlist_tracks(my_playlist_URI)

    song_uri = []
    song_name = []
    artist = []
    artist_main_genre = []
    song_popularity = []

    for entry in tracks_in_my_playlist_info["items"]:
        song_uri.append(entry["track"]["uri"].split(":")[-1])
        song_name.append(entry["track"]["name"])
        artist.append(entry["track"]["artists"][0]["name"])

        try: 
            artist_info = sp.artist(entry["track"]["artists"][0]["uri"])
            artist_genres = artist_info["genres"]
            artist_main_genre.append(artist_genres[0] if artist_genres else "unknown")
        except IndexError:
            artist_main_genre.append("unknown")
        song_popularity.append(entry["track"]["popularity"])

    basic_song_data = pd.DataFrame({'song_uri' : song_uri, 'song_name' : song_name, 'artist' : artist, 'genre' : artist_main_genre, 'popularity' : song_popularity})


    detailed_song_features = sp.audio_features(basic_song_data['song_uri'])
    detailed_song_features = pd.DataFrame.from_dict(detailed_song_features)


    all_songs_data_for_my_playlist = pd.merge(left = basic_song_data, right = detailed_song_features, left_on = "song_uri", right_on = "id")

    simple_song_data_for_my_playlist = all_songs_data_for_my_playlist[["song_name", "artist", "genre", "popularity", "danceability", "energy", "acousticness", "instrumentalness", "liveness", "valence", "duration_ms"]]

    return simple_song_data_for_my_playlist


train_data = get_playlist_data('my_playlist_link')
print(train_data.head())

def basic_scatter(df, column_x, column_y):
    plt.scatter(x = df[column_x], y=df[column_y])
    plt.title("Megan's October Playlist Metrics")
    plt.xlabel(column_x)
    plt.xlim([0,1])
    plt.ylabel(column_y)
    plt.ylim([0,1]);

#basic_scatter(train_data, "danceability", "valence")


def elbow_method(df, column_x, column_y):
    wcss=[]

    for i in range(1,11):
        kmeans= KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(df[[column_x, column_y]])

        wcss.append(kmeans.inertia_)

    plt.plot(range(1,11),wcss)
    plt.title("Clustering")
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia');

elbow_method(train_data, "danceability", "valence")

def kmeans_training_from_scratch(n_clusters, df, column_x, column_y)
    initial_cluster_centroids = df[[column_x, column_y]].sample(n_clusters)
    initial_cluster_centroids.reset_index(inplace=True, drop=True)

    index_of_closest_cluster = []

    for song_index, song_row in df[[column_x, column_y]].iterrows():

        distances = {}

        for centroid_index, centroid_row in initial_cluster_centroids.iterrows():
            individual_x_distance = song_row[column_x] - centroid_row[column_x]
            individual_y_distance = song_row[column_y] - centroid_row[column_y]
            distances[centroid_index] = np.sqrt(individual_x_distance**2 + individual_y_distance**2)

        index_of_closest_cluster.append(min(distances, key=distances.get))

    df["Previous Cluster"] = index_of_closest_cluster

    previous_clustering = df.groupby(
        df["Previous Cluster"]).mean()[[column_x, column_y]]
    previous_clustering = previous_clustering.sort_values(by = column_x).reset_index(drop= True)

    fig, ax = plt.subplots(2, figsize = (6,12))
    for n in range(0, n_clusters):
        ax[0].scatter(df[df["Previous Cluster"] == n][column_x]
                    df[df["Previous Cluster"] == n][column_y],
                    label = "Cluster " + str(n+1))    
        
    ax[0].scatter(x =initial_cluster_centroids[column_x], 
                y = initial_cluster_centroids[column_y], 
                c = "red", s = 200,
                label = 'Centroids')
    ax[0].set_title("Metrics for Playlist with Initial Nonrandom Clusters")
    ax[0].set_xlabel(column_x)
    ax[0].set_xlim([0, 1])
    ax[0].set_ylabel(column_y)
    ax[0].set_ylim([0, 1])
    ax[0].legend();
        

    iterations = 0
    max_iter = 300


    while iterations <= max_iter:

        
        index_of_closest_cluster = []
        for song_index, song_row in df[[column_x, column_y]].iterrows():
            distances = {}
            for centroid_index, centroid_row in previous_clustering.iterrows():
                individual_x_distance = song_row[column_x] - centroid_row[column_x]
                individual_y_distance = song_row[column_y] - centroid_row[column_y]
                distances[centroid_index] = np.sqrt(individual_x_distance**2 + individual_y_distance**2)
            index_of_closest_cluster.append(min(distances, key=distances.get))
        df["Current Cluster"] = index_of_closest_cluster
        current_clustering = df.groupby(
            df["Current Cluster"]).mean()[[column_x, column_y]]
        current_clustering = current_clustering.sort_values(by = column_x).reset_index(drop = True)
        
      
        iterations += 1
        
    
        if df["Previous Cluster"].equals(df["Current Cluster"]) | previous_clustering.sort_values(by = column_x).equals(current_clustering.sort_values(by = column_x)):
            
          
            df.drop(columns = ["Previous Cluster"], inplace = True)
            iterations -= 1
            print("The model has been trained in", iterations, "iterations")
            iterations = 301
        
       
        else:
            df["Previous Cluster"] = df["Current Cluster"]
            previous_clustering = current_clustering

   
    for n in range(0, n_clusters):
        ax[1].scatter(df[df["Current Cluster"] == n][column_x], 
                    df[df["Current Cluster"] == n][column_y], 
                    label = "Cluster " + str(n + 1))

    ax[1].scatter(current_clustering[column_x],
                current_clustering[column_y],
                c = "red", s = 200,
                label = 'Centroids')
    ax[1].set_title("Metrics for Playlist with Final Nonrandom Clusters")
    ax[1].set_xlabel(column_x)
    ax[1].set_xlim([0, 1])
    ax[1].set_ylabel(column_y)
    ax[1].set_ylim([0, 1])
    ax[1].legend();