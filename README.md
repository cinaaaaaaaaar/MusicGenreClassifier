# Stats

I've divided the audio files from the [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) into 10 segments and then fed them to the network. The final accuracy was %82 and the training took about 2 minutes 5 seconds to finish. The model overfitted in my first tries, but after some hyperparameter tweaking I've partially solved overfitting.

![Accuracy and Loss Plots](https://raw.githubusercontent.com/ozlucinar/genre_classifier/main/images/accuracy_loss_plot.png)
