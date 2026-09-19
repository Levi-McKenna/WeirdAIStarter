from weird_ai.dataset import LyricsDataset


def test_lyrics_dataset_getitem():
    tokens = [1, 2, 3, 4, 5]
    dataset = LyricsDataset(tokens, block_size=3, stride=1)

    x, y = dataset[0]

    assert list(x) == [1, 2, 3]
    assert list(y) == [2, 3, 4]
