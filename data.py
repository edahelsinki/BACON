from pathlib import Path

import numpy as np
import pandas as pd
import sklearn as sklearn
from clustpy.data import load_bbcnews, load_bbcsport, load_optdigits
from clustpy.data.real_torchvision_data import (
    load_mnist,
    load_usps,
)
from clustpy.data.real_uci_data import (
    load_reuters21578,
)
from clustpy.deep.neural_networks.feedforward_autoencoder import (
    FeedforwardAutoencoder,
)
from sklearn.datasets import (
    load_breast_cancer,
    make_moons,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo


def get_data_path():
    # create a results dir
    data_dir = Path(__file__).parent / "data"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    return data_dir


def standardize(X, normalizing_const=None):
    if normalizing_const is None:
        # use z-scaling
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        # normalize by dividing by a constant
        X = X / normalizing_const
    return X


def embed_and_save(X, y, x_path, y_path):
    ae = FeedforwardAutoencoder(layers=[X.shape[1], 1000, 500, 500, 10], random_state=42)
    ae.fit(data=X)
    Z = ae.transform(X)
    # Z = standardize(Z)
    np.save(x_path, Z)
    np.save(y_path, y)
    return Z, y


def get_optdigits(normalize=True):
    # clusters = 10, d = 64
    data_dir = get_data_path()
    x_path = data_dir / "optdigits_x.npy"
    y_path = data_dir / "optdigits_y.npy"
    try:
        X = np.load(x_path)
        y = np.load(y_path)
    except(FileNotFoundError):
        print("Trying to embed optdigits")
        X, y = load_optdigits(return_X_y=True, downloads_path=data_dir)
        X = standardize(X, normalizing_const=16.0)
        X, y = embed_and_save(X, y, x_path, y_path)

    return X, y 


def get_usps(normalize=True):
    # clusters = 10, d = 256
    data_dir = get_data_path()
    x_path = data_dir / "usps_x.npy"
    y_path = data_dir / "usps_y.npy"
    try:
        X = np.load(x_path)
        y = np.load(y_path)
    except(FileNotFoundError):
        print("Trying to embed usps")
        X, y = load_usps(return_X_y=True, downloads_path=data_dir)
        X = standardize(X, normalizing_const=255.0)
        X, y = embed_and_save(X, y, x_path, y_path)

    return X, y


def get_mnist(normalize=True):
    data_dir = get_data_path()
    x_path = data_dir / "mnist_x.npy"
    y_path = data_dir / "mnist_y.npy"
    try:
        X = np.load(x_path)
        y = np.load(y_path)
    except(FileNotFoundError):
        print("Trying to embed usps")
        X, y = load_mnist(return_X_y=True, downloads_path=data_dir)
        X = standardize(X, normalizing_const=255.0)
        X, y = embed_and_save(X, y, x_path, y_path)

    return X, y


def get_reuters(normalize=True):
    data_path = get_data_path()
    x_path = data_path / "reuters_x.npy"
    y_path = data_path / "reuters_y.npy"
    try:
        X = np.load(x_path)
        y = np.load(y_path)
    except FileNotFoundError:
        print("Trying to embed reuters")
        X, y = load_reuters21578(return_X_y=True)
        X, y = embed_and_save(X, y, x_path, y_path)
    return X, y


def get_bbcsports(normalize=True):
    data_path = get_data_path()
    x_path = data_path / "bbcsports_x.npy"
    y_path = data_path / "bbcsports_y.npy"
    try:
        X = np.load(x_path)
        y = np.load(y_path)
    except FileNotFoundError:
        print("Trying to embed bbcsports")
        X, y = load_bbcsport(return_X_y=True)
        X, y = embed_and_save(X, y, x_path, y_path)

    return X, y


def get_bbcnews(normalize=True):
    data_path = get_data_path()
    x_path = data_path / "bbcnews_x.npy"
    y_path = data_path / "bbcnews_y.npy"
    try:
        X = np.load(x_path)
        y = np.load(y_path)
    except FileNotFoundError:
        print("Trying to embed bbcnews")
        X, y = load_bbcnews(return_X_y=True)
        X, y = embed_and_save(X, y, x_path, y_path)

    return X, y

def get_moons(n_samples=100, normalize=True):
    X, y = make_moons(n_samples=n_samples, noise=0.05, random_state=42)
    X = standardize(X)
    return X, y

def get_iris(normalize=True):
    X, y = sklearn.datasets.load_iris(return_X_y=True)
    X = standardize(X)
    return X, y

def get_wine(normalize=True):
    X, y = sklearn.datasets.load_wine(return_X_y=True)
    X = standardize(X)
    return X, y


def get_waveform(normalize=True):
    # clusters = 3, d = 21
    waveform = fetch_ucirepo(id=107) 
    # data (as pandas dataframes) 
    X = waveform.data.features 
    y = waveform.data.targets
    X = X.to_numpy()
    y = y.to_numpy().flatten()
    X = standardize(X)
    return X, y


def get_segmentation(normalize=True):
    data_dir = get_data_path() / "image+segmentation"
    df = pd.concat([pd.read_csv(data_dir / "segmentation.data.txt"), pd.read_csv(data_dir / "segmentation.test.txt")])
    X = df.to_numpy()
    y = df.index.to_numpy()
    labels = np.zeros(len(y), dtype=int)
    labels[y == "SKY"] = 1
    labels[y == "FOLIAGE"] = 2
    labels[y == "CEMENT"] = 3
    labels[y == "WINDOW"] = 4
    labels[y == "PATH"] = 5
    labels[y == "GRASS"] = 6
    y = labels
    X = standardize(X)
    return X, y


def get_dermatology(normalize=True):
    # fetch dataset 
    dermatology = fetch_ucirepo(id=33) 
    # data (as pandas dataframes) 
    data = dermatology.data.features
    target = dermatology.data.targets
    # remove nans
    nan_mask = ~data.isna().any(axis=1)
    X = data[nan_mask].to_numpy()
    y = target[nan_mask].to_numpy().flatten() - 1
    X = standardize(X)
    return X, y


def get_glass(normalize=True):
    # fetch dataset 
    glass_identification = fetch_ucirepo(id=42) 
    # data (as pandas dataframes) 
    data = glass_identification.data.features
    target = glass_identification.data.targets
    X = data.to_numpy()
    y = target.to_numpy().flatten()
    # make targets 0-5
    y[y==6] = 0
    y[y==7] = 4
    X = standardize(X)
    return X, y


def get_spambase(normalize=True):
    spambase = fetch_ucirepo(id=94)
     # data (as pandas dataframes) 
    data = spambase.data.features
    target = spambase.data.targets
    X = data.to_numpy()
    y = target.to_numpy().flatten()
    X = standardize(X)
    return X, y


def get_parkinsons(normalize=True):
    parkinsons = fetch_ucirepo(id=174) 
    X = parkinsons.data.features.to_numpy()
    y = parkinsons.data.targets.to_numpy().flatten()
    X = standardize(X)
    return X, y


def get_ecoli(normalize=True):
    ecoli = fetch_ucirepo(id=39) 
    X = ecoli.data.features.to_numpy()
    labels = ecoli.data.targets.to_numpy().flatten()
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    X = standardize(X)
    return X, y


def get_breast_cancer(normalize=True):
    X, y = load_breast_cancer(return_X_y=True)
    X = standardize(X)
    return X, y



if __name__ == "__main__":
    datasets = [
        ("Moons", get_moons),
        ("Iris", get_iris),
        ("Wine", get_wine),
        ("Dermatology", get_dermatology),
        ("Ecoli", get_ecoli),
        ("Glass", get_glass),
        ("Parkinsons", get_parkinsons),
        ("BreastCancer", get_breast_cancer),
        ("Segmentation", get_segmentation),
        ("Waveform", get_waveform),
        ("Optdigits", get_optdigits),
        ("MNIST", get_mnist),
        ("USPS", get_usps),
        ("Reuters", get_reuters),
        ("BBCNews", get_bbcnews),
        ("BBCSports", get_bbcsports),
    ]

    print("Loading and summarizing datasets ...")
    print("====================================")
    for datasets_name, dataset_loader in datasets:
        X, y = dataset_loader()
        n, d = X.shape
        k = len(np.unique(y))
        print(f"{datasets_name} & {n} & {k} & {d} \\\\")
        # print()
    print("====================================")
