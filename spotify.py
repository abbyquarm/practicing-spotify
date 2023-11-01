
from getpass import getpass
import os
from dotenv import load_dotenv
load_dotenv()

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

client_credentials_manager = SpotifyClientCredentials(client_id = CLIENT_ID, client_secret = CLIENT_SECRET)

sp = spotipy.Spotify(client_credentials_manager= client_credentials_manager)

my_playlist_link ='https://open.spotify.com/playlist/3rMrPHuQFD4ihqR7fWZaSQ'
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

print(simple_song_data_for_my_playlist)





