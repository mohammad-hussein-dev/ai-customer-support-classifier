"""Unit tests for feature engineering."""

from scipy.sparse import csr_matrix

from src.features.build_features import FeatureBuilder


class TestFeatureBuilder:
    """Test suite for FeatureBuilder."""

    def test_initialization(self):
        builder = FeatureBuilder()
        assert builder.max_features == 10000
        assert builder.ngram_range == (1, 2)

    def test_fit_transform(self):
        texts = [
            "hello world test",
            "hello python code",
            "test machine learning",
        ]
        builder = FeatureBuilder(max_features=100)
        matrix = builder.fit_transform(texts)
        assert isinstance(matrix, csr_matrix)
        assert matrix.shape[0] == 3
        assert matrix.shape[1] <= 100

    def test_vocabulary_size(self):
        texts = ["hello world", "hello test"]
        builder = FeatureBuilder(max_features=10)
        builder.fit(texts)
        assert len(builder.get_feature_names()) > 0

    def test_transform_new_text(self):
        texts = ["hello world", "python code"]
        builder = FeatureBuilder(max_features=10)
        builder.fit(texts)
        new_matrix = builder.transform(["hello python"])
        assert new_matrix.shape[0] == 1
