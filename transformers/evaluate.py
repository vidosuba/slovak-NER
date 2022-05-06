import sys

sys.path.append('../utils')
from evaluate_ner import evaluate_with_gold, evaluate_with_wikiann, evaluate_with_wikiann_hg


def main():
    predictions_path = 'predictions.txt'

    with open(predictions_path) as f:
        content = f.read().strip()

    predictions_sentences = content.split('\n')
    predictions = [p_s.split() for p_s in predictions_sentences]

    evaluate_with_gold(predictions)
    #evaluate_with_wikiann(predictions)
    #evaluate_with_wikiann_hg(predictions)


if __name__ == '__main__':
    main()