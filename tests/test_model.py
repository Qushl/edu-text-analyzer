from src.model import train_model


def test_model_training():
    model, vectorizer = train_model("data/sample_data.csv")
    assert model is not None
    assert vectorizer is not None
