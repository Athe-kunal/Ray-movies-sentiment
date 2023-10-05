import ray
from data import *
ray.data.DatasetContext.get_current().execution_options.preserve_order = True

def get_train_data(DATASET_LOC:str,val_size:float=0.2):
    ds = load_data(dataset_loc=DATASET_LOC)
    train_ds, val_ds = stratify_split(ds, stratify="Ratings_Col", test_size=val_size)

    preprocessor = CustomPreprocessor()
    preprocessor = preprocessor.fit(train_ds)
    train_ds = preprocessor.transform(train_ds)
    val_ds = preprocessor.transform(val_ds)
    train_ds = train_ds.materialize()
    val_ds = val_ds.materialize()

    return train_ds,val_ds, preprocessor

