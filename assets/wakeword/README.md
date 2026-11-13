# Training the "Hey Zeno" wake word

The wake word is a small neural network (openWakeWord) trained **off this repo** in a free Google
Colab notebook, because the training code needs PyTorch and Linux. Only the finished model,
`hey_zeno.onnx` (a few hundred KB), comes back here.

openWakeWord learns from *synthetic* speech: the notebook generates thousands of "Hey Zeno" clips
with text-to-speech, mixes in background noise, and trains against about 2,000 hours of
non-wake-word audio. No recording is needed for training. Your own recordings are only used
afterwards, to **test** the model.

**Time:** about 2 to 3 hours, mostly waiting, with the Colab tab kept open. I have not been able to
run this notebook myself, so treat times and a few details below as estimates.

## 1. Open the notebook in Colab

1. Go to https://colab.research.google.com/github/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb
   (the notebook `automatic_model_training.ipynb` from the openWakeWord repository). Sign in with your Google account.
2. **Runtime > Change runtime type > T4 GPU**, then Save.
3. Keep the tab open and your laptop awake until training finishes. The free tier may disconnect if idle.

## 2. Run it, with two changes

Run the cells **in order, top to bottom** (click the play button on each, wait for it to finish):

1. **Environment setup** (the cell that starts with `# install piper-sample-generator`). This installs everything.
2. The imports cell.
3. The cells that **download data**: AudioSet background noise, room impulse responses, and the
   pre-computed features. The features file is **17.3 GB**. That is fine on Colab's fast connection (it
   takes a while) and is the reason this is not done on your laptop.
4. The cell that loads the default config (`config = yaml.load(...)`).
5. **Change the config cell.** Replace the cell that sets `config["target_phrase"] = ["hey sebastian"]` with:

   ```python
   config["target_phrase"] = ["hey zeno", "hey zeeno"]
   config["model_name"] = "hey_zeno"
   config["custom_negative_phrases"] = [
       "hey zoe", "he knows", "see no evil", "hey jenny", "say no more", "hey siri", "okay google", "hey zero",
   ]
   config["n_samples"] = 10000
   config["n_samples_val"] = 2000
   config["steps"] = 30000
   config["target_accuracy"] = 0.6
   config["target_recall"] = 0.25

   config["background_paths"] = ['./audioset_16k', './fma']
   config["false_positive_validation_data_path"] = "validation_set_features.npy"
   config["feature_data_files"] = {"ACAV100M_sample": "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"}

   with open('my_model.yaml', 'w') as file:
       documents = yaml.dump(config, file)
   ```
   Keep the last three lines of the cell the same as the notebook's version (if the notebook's cell
   differs slightly from this one, keep its structure and only change the values above).
   *"hey zeno" and "hey zeeno" both give the pronunciation ZEE-noh. Don't add "zino": it would train the "ZY-noh" sound.*
6. Run **Step 1** (generate clips), **Step 2** (augment clips), **Step 3** (train model) in that order.
   If a step fails part-way, run the same cell again. Generation resumes where it stopped.

## 3. Download the model

When Step 3 finishes, the file is `my_custom_model/hey_zeno.onnx`. Download it:

- In Colab's left sidebar open the **Files** panel (folder icon), open `my_custom_model`, right-click
  `hey_zeno.onnx`, and choose **Download**. Or run this in a new cell:

  ```python
  from google.colab import files
  files.download("my_custom_model/hey_zeno.onnx")
  ```

Save it as **`assets/wakeword/hey_zeno.onnx`** in this repo.

## 4. Test it on your own voice

```powershell
# Does it load, and does the synthetic test clip fire?
.\.venv\Scripts\python.exe -m core.wake --file tests\fixtures\audio\hey_zeno.wav --model hey_zeno

# Record 20 "Hey Zeno" clips and 20 other phrases (about 5 minutes; speak naturally, vary distance and volume)
.\.venv\Scripts\python.exe -m core.wake --record

# Record 10 minutes of the TV, music or room noise you usually have on, and talk now and then
.\.venv\Scripts\python.exe -m core.wake --record-bg 10

# Score everything and show a threshold table
.\.venv\Scripts\python.exe -m core.wake --eval --model hey_zeno
```

**Goal** (from TASKS.md): at the chosen threshold, at least 90% of your "Hey Zeno" clips fire, none of
the other phrases fire, and the background recording gives under 1 false trigger per hour.
Your recordings are saved in `data/wakeword/`, which is git-ignored. Never commit them.

## If something goes wrong

- **The setup cell fails** on a package (the notebook pins old versions such as `tensorflow-cpu==2.8.1`,
  and Colab's Python keeps moving on). Only the ONNX model is needed here, so copy the full error message
  and ask for help. Many of those pins are only for the optional `.tflite` conversion and can be skipped.
- **Detection is below 90%** on your voice: retrain with `n_samples = 20000` and `steps = 50000`. If that
  still isn't enough, the next step is a *custom verifier* (a small classifier on your own recordings,
  in openWakeWord's `docs/custom_verifier_models.md`). It works well but only for the voices it was trained on.
- **Too many false triggers:** raise the threshold in `config.yaml` (`wake.threshold`), add the offending
  phrases to `custom_negative_phrases`, and retrain.

## Licensing

The openWakeWord code is Apache-2.0, but its **pre-trained models** (hey_jarvis and friends) are
CC BY-NC-SA 4.0 (non-commercial), and the README says its training data has "unknown or restrictive
licensing". Zeno is for personal use and is not sold, so this is fine. Those pre-trained models are used
only for development and tests: they are downloaded at run time and **never committed**. If Zeno is
ever distributed commercially, `hey_zeno.onnx` needs a licensing review first.
