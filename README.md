
# DCASE 2019: Sound event localization and detection (SELD) task 3

This is a demonstrative repo of a few models submitted to the DCASE2019 Challenge, task 3 (SELD).

The system is based upon the baseline provided by the organizers (see: https://github.com/sharathadavanne/seld-dcase2019) [1][2] and works due to it's documentation.

Three main submitted systems are presented:

-Submission system 1 (keras_model1.py)

-Submission system 2 (keras_model2.py)

-Submission system 3 (keras_model3.py)


If you want to run the training for any of these models, use the "model_nb" parameter in parameter.py.

Also a fourth system has been submitted, however it's just a simple ensemble of the first two models. Therefore in order to test it, you should train the first two models separately
and then load these models in the testing process and test them as an ensemble.

parameter.py, calculate_SELD_metrics.py and batch_feature_extraction.py contain dir variables. You need to fix them depending on where you've extracted the data.

[1] Sharath Adavanne, Archontis Politis, Joonas Nikunen and Tuomas Virtanen, "Sound event localization and detection of overlapping sources using convolutional recurrent neural network" in IEEE Journal of Selected Topics in Signal Processing (JSTSP 2018)

[2] Sharath Adavanne, Archontis Politis, Joonas Nikunen and Tuomas Virtanen, "A multi-room reverberant dataset for sound event localization and detection" submitted in the Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE 2019)

## License

Except for the contents in the `metrics` folder that have [MIT License](metrics/LICENSE.md). The rest of the repository is licensed under the [TAU License](LICENSE.md).