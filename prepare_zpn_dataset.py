import pandas as pd

def column_to_zpn(column):
    mean = column.mean()
    std = column.std()
    encoder = [-1, 0, 1]
    bounds = [mean-std/2, mean+std/2]
    encoded_column = []
    for input in column:
        encoded_column.append(encoder[next((i for i, x in enumerate(bounds) if input <= x), len(bounds))])
    return encoded_column


def prepare_zpn_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path)
    dataset.columns = [f'dim_{idx}' for idx in range(len(dataset.columns) - 1)] + ['class']
    for column in dataset.columns[:-1]:
        dataset[column] = column_to_zpn(dataset[column])
    dataset.to_csv('letter_zpn.csv', index=False, header=False)


if __name__ == '__main__':
    prepare_zpn_dataset('/home/pawel/PycharmProjects/DAMI_project/datasets/letter.csv')