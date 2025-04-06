# 🎤 Speech-to-Text Transcriber

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aishwaryaj7-speech2text.streamlit.app/)


A simple and powerful Streamlit web app that transcribes speech from audio/video files using state-of-the-art transformer models from Hugging Face 🤗.

Watch a demo in the app, upload a file, and receive a clean, downloadable transcript — all in your browser!

![Streamlit](https://img.shields.io/badge/Streamlit-1.44.0+-brightgreen?logo=streamlit)
![Transformers](https://img.shields.io/badge/Transformers-4.50.3+-blueviolet?logo=python)
![Torchaudio](https://img.shields.io/badge/torchaudio-2.6.0+-red)
![HuggingFace Hub](https://img.shields.io/badge/huggingface--hub-0.30.1+-yellow?logo=huggingface)
![YouTube-dl](https://img.shields.io/badge/youtube--dl-2021.12.17+-black)

---

## 🚀 Features

- 🎙️ Upload audio/video in `.mp3`, `.mp4`, `.wav`
- 🧠 Transcribe using `facebook/hubert-large-ls960-ft`
- 📝 View & download transcripts
- 🎞️ Watch a demonstration video directly in the app
- 📋 Step-by-step view of timestamped segments

---

## 🧰 Tech Stack

- 🤗 Transformers + Hubert model for CTC-based transcription
- 🧠 Wav2Vec2Processor tokenizer
- 📦 Streamlit for interactive UI
- 🎧 Torchaudio + Librosa + Pydub for preprocessing

### App Demo

[![Video](https://img.youtube.com/vi/eSG_FsoUtRo/hqdefault.jpg)](https://www.youtube.com/watch?v=eSG_FsoUtRo)

