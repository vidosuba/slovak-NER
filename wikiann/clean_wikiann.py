import re


def process_row(row, conll_2003=False):
    cleaned = re.sub(r'^sk:', '', row)
    splitted = cleaned.split()
    if len(splitted) <= 0:
        return
    text = re.sub(r'[^\w\s.,!?]', '', splitted[0])
    tag = splitted[1]
    if len(text) > 0:
        if conll_2003:
            return ' '.join([text, '.', 'O', tag])
        else:
            return ' '.join([text, tag])


def clean_wiki(in_path, out_path):
    with open(in_path) as in_f:
        with open(out_path, 'w') as out_f:
            content = in_f.read()
            for chunk in content.split('\n\n'):
                for row in chunk.split('\n'):
                    cleaned = process_row(row)
                    if cleaned is not None:
                        print(cleaned, file=out_f)
                print('', file=out_f)


def main():
    datasets = ['test', 'dev', 'train']
    for dataset in datasets:
        in_path = f'raw_data/{dataset}.txt'
        out_path = f'cleaned_data/{dataset}_cleaned.txt'
        clean_wiki(in_path, out_path)


main()
