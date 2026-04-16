# Deepfake-AI-Voice-Detection-Model

## How to train each model
- unzip the audio files in 10s_clips/AUDIO_CLEAN
- Go to Indiviual Model directory
- run Audio_to_mel.py to create dataset
- run each model to train and build weights
- move the weights from best_weights folder to inference/Voice-Ai folder to be use in ensemble model

## Turn on Web inference 
- Go to Inference/Webpage
- type npm run dev in console
- go to Inference/Voice-Ai 
- type uvincorn ensamble:app in console