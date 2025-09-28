import streamlit as st
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import io

def extract_features(audio_file):
    y, sr = librosa.load(audio_file, duration=30)
    rms_energy = librosa.feature.rms(y=y)[0]
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return rms_energy, tempo, beats, mfccs_mean, sr

def plot_energy_beat(rms_energy, beats, sr):
    fig, ax = plt.subplots(figsize=(10, 4))
    times = librosa.times_like(rms_energy, sr=sr)
    ax.plot(times, rms_energy, label='RMS Energy', color='#88d498')
    ax.set_title('Audio Beat Tracker Visualization', fontsize=16)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RMS Energy')
    beat_times = librosa.frames_to_time(beats, sr=sr)
    ax.vlines(beat_times, 0, np.max(rms_energy), color='r', linestyle='--', alpha=0.5, label='Beat Onsets')
    plt.close(fig)
    return fig

def plot_spectrogram(y, sr, mood_label, mood_color):
    fig, ax = plt.subplots(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax, cmap='magma')
    ax.set_title(f'Frequency Spectrogram (Mood: {mood_label})', fontsize=16, color=mood_color)
    ax.tick_params(labelcolor='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor(mood_color)
    plt.close(fig)
    return fig

DUMMY_DATA = np.array([
    [0.5, 0.1, 0.3], [0.6, 0.2, 0.4], [0.1, 0.9, 0.8], [0.2, 0.8, 0.9],
    [0.9, 0.05, 0.1], [0.8, 0.1, 0.05], [0.15, 0.85, 0.85], [0.05, 0.95, 0.9]
])
DUMMY_LABELS = {0: ('Energetic', 'red'), 1: ('Chill', 'blue'), 2: ('Neutral', 'green')}
MOOD_MODEL = KMeans(n_clusters=3, random_state=42, n_init=10)
MOOD_MODEL.fit(DUMMY_DATA)

def classify_mood(mfccs_mean):
    mfccs_subset = mfccs_mean[:3].reshape(1, -1)
    prediction = MOOD_MODEL.predict(mfccs_subset)[0]
    return DUMMY_LABELS.get(prediction, ('Unknown', 'gray'))

def main():
    st.set_page_config(layout="wide", page_title="AI Music Visualizer")
    st.title("AI-Powered Music Visualizer 🎶")
    st.markdown("Upload an audio file to see its energy, beat, and AI-predicted mood.")
    
    audio_file = st.file_uploader("Upload Audio File (MP3/WAV)", type=["mp3", "wav"])
    
    if audio_file:
        audio_bytes = audio_file.read()
        audio_buffer = io.BytesIO(audio_bytes)
        
        st.audio(audio_bytes, format='audio/wav')
        
        with st.spinner('Analyzing audio features...'):
            rms_energy, tempo, beats, mfccs_mean, sr = extract_features(audio_buffer)
            mood_label, mood_color = classify_mood(mfccs_mean)
        
        st.success(f"Analysis Complete! Tempo: **{int(tempo)} BPM**")
        
        st.markdown("---")
        st.header(f"AI Mood Classification: :bulb: <span style='color:{mood_color}'>{mood_label}</span>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Beat Tracking and Energy Profile")
            fig_beat = plot_energy_beat(rms_energy, beats, sr)
            st.pyplot(fig_beat)
            st.markdown("*(Vertical lines show detected beat onsets)*")
            
        with col2:
            st.subheader("Frequency Spectrogram")
            y_spec, sr_spec = librosa.load(audio_buffer, duration=30)
            fig_spec = plot_spectrogram(y_spec, sr_spec, mood_label, mood_color)
            st.pyplot(fig_spec)
            st.markdown("*(Visualizes frequency changes over time)*")
            
if __name__ == "__main__":
    main()
