import pandas as pd

from embeddings.vit_finetuning import ProductDataset


def make_df(categories):
    return pd.DataFrame({
        "category": categories,
        "image_filename": [f"img_{i}.jpg" for i in range(len(categories))],
    })


def test_categories_are_sorted_and_deduplicated():
    df = make_df(["Toys", "Electronics", "Toys", "Beauty", "Electronics"])
    dataset = ProductDataset(df, transform=None)
    assert dataset.categories == ["Beauty", "Electronics", "Toys"]


def test_cat_to_idx_maps_each_category_to_unique_index():
    df = make_df(["Toys", "Electronics", "Beauty"])
    dataset = ProductDataset(df, transform=None)
    indices = list(dataset.cat_to_idx.values())
    assert len(indices) == len(set(indices))
    assert set(indices) == {0, 1, 2}


def test_mapping_is_deterministic_regardless_of_row_order():
    df1 = make_df(["Toys", "Electronics", "Beauty"])
    df2 = make_df(["Beauty", "Toys", "Electronics"])
    dataset1 = ProductDataset(df1, transform=None)
    dataset2 = ProductDataset(df2, transform=None)
    assert dataset1.cat_to_idx == dataset2.cat_to_idx


def test_len_matches_dataframe_row_count():
    df = make_df(["Toys", "Electronics", "Beauty", "Toys"])
    dataset = ProductDataset(df, transform=None)
    assert len(dataset) == 4


def test_single_category_dataset():
    df = make_df(["Electronics", "Electronics", "Electronics"])
    dataset = ProductDataset(df, transform=None)
    assert dataset.categories == ["Electronics"]
    assert dataset.cat_to_idx == {"Electronics": 0}
