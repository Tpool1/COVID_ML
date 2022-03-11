from src.data_pipeline import data_pipeline
from src.cnn import cnn

class covid:
    def __init__(self):
        pipe = data_pipeline("COVID-19-Data\\deidentified_overlap_tcia.csv.cleaned.csv_20210806.csv", None, "COVID-19-Data\\img_array_covid.npy", "last.status")
        pipe.load_data()

        model = cnn(load_model=False)
        model.train_model(pipe.image_only.X_train, pipe.image_only.y_train, pipe.image_only.X_val, pipe.image_only.y_val)
        model.test_model()

classifier = covid()
