from transformers import pipeline
from spacy import displacy
import nltk
import jsonlines
import textspan
from tqdm import tqdm


nltk.download('punkt')


def visualize_ner(ner_data, out_path='visualized_ner.html'):
    '''Generates HTML from NER data'''
    html = displacy.render(ner_data, options={'colors': {'PER': 'yellow', 'MISC': '#f89ff8'}}, manual=True, style="ent", page=True)
    with open(out_path, 'w') as f:
        f.write(html)


def visualize_prodigy(annotations_prodigy):
    '''Visualize Spacy/Prodigy NER data'''
    annotations_prodigy['ents'] = annotations_prodigy['spans']
    visualize_ner(annotations_prodigy, out_path='visualized_ner_prodigy.html')


def tokenize_nltk(text):
    '''Tokenize article to sentences and words'''
    tokenized = []
    sentence_tokenizer = nltk.data.load(f"tokenizers/punkt/czech.pickle")
    extra_abbreviations = ['napr', 'lat', 'rod', 'mr', 'sv', 'mgr',
                           'prof', 'ing', 'bc', 'dr', 'rus', 'tzv', 'phd', 'drsc', 'phdr',
                           'dr', 'iii', 'ii', 'i', 'iv', 'odd', 'angl', 'skr', 'stor',
                           'pol', 'vz', 'tal', 'rndr', 'fr', 'odd', 'mad', 'var', 'grec',
                           'gr', 'nem', 'lat', 'hebr', 'arab', 'novoheb', ]
    sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)
    sentences = sentence_tokenizer.tokenize(text)

    for sentence in sentences:
        tokenized.append(nltk.tokenize.word_tokenize(sentence, language='czech', preserve_line=True))
    return tokenized


def tokenize_split(text):
    sentences = text.split(' . ')
    return [sentence.split(' ') for sentence in sentences]


def join_tokenized_nltk(tokenized_text):
    return ' '.join([' '.join(sentence) for sentence in tokenized_text])


def connect_annotations(annotations, sentence):
    '''Map and align annotations from transformer model with tokens from NLTK'''
    annotations_s = sorted(annotations, key=lambda x: x['start'])
    connected = []
    tokens = sentence.split(' ')

    annotation_ix = 0
    token_ix = -1
    tokens_len = -1
    tokens_j = ''

    while annotation_ix < len(annotations_s):
        current_annotation = annotations_s[annotation_ix]

        while tokens_len < current_annotation['end']:
            token_ix += 1
            tokens_len += len(tokens[token_ix]) + 1
            tokens_j = tokens_j + tokens[token_ix] + ' '

        if current_annotation['end'] != tokens_len:

            new_annotation = dict(current_annotation)

            while True:
                annotation_ix += 1
                if annotation_ix >= len(annotations_s):
                    break
                current_annotation = annotations_s[annotation_ix]

                new_annotation = {
                    'text': sentence[new_annotation['start']:current_annotation['end']],
                    'label': new_annotation['label'],
                    'start': new_annotation['start'],
                    'end': current_annotation['end'],
                }

                if tokens_len <= current_annotation['end']:
                    break

            connected.append(new_annotation)
        else:
            connected.append(current_annotation)

        annotation_ix += 1

    return connected


def filter_annotations(annotations, tokenized_sentence):
    '''Filter out annotations that are mostly mistakes'''
    # remove 'headers like' annotations - one word and dot
    if len(tokenized_sentence) == 2 and tokenized_sentence[1] == '.':
        return []

    # remove annotations that don't start with capital letter
    return [ann for ann in annotations if ann['text'][0].isupper()]


def get_clear_annotations(annotations_encoded, text):
    '''Transform annotations from transformer model to more clear dict format'''
    annotations = []
    for annotation in annotations_encoded:
        annotations.append({
            'text': text[annotation['start']:annotation['end']],
            'label': annotation['entity_group'],
            'start': annotation['start'],
            'end': annotation['end'],
        })
    return annotations


def inference(pipeline, tokenized_sentence):
    '''Predict, transform and filter NER labels for a sentence'''
    joined_sentence = ' '.join(tokenized_sentence)
    annotations = pipeline(joined_sentence)
    clear_annotations = get_clear_annotations(annotations, joined_sentence)
    connected_annotations = connect_annotations(clear_annotations, joined_sentence)
    filtered_annotations = filter_annotations(connected_annotations, tokenized_sentence)

    return filtered_annotations


def inference_wikidump(dataset_path, pipeline):
    '''Get predicted annotations for jsonl wikidump dataset'''
    n_documents = 99999
    wikidump_labeled = []

    num_lines = sum(1 for line in open(dataset_path))

    with jsonlines.open(dataset_path) as dataset_file:
        document_count = 0
        print(f'WIKIANN SLOVAK BERT INFERENCING {dataset_path}')
        for doc in tqdm(dataset_file, total=num_lines):

            ner_data = []
            tokenized_text = tokenize_nltk(doc['text'])
            for sentence in tokenized_text:
                annotations = inference(pipeline, sentence)
                ner_data.append({
                    'text': ' '.join(sentence),
                    'ents': annotations,
                })

            document_count += 1
            wikidump_labeled.append({
                'sentences': ner_data,
                'meta': doc['meta'],
            })

            if document_count >= n_documents:
                break
    #visualize_ner({'text': ' '.join(tokenized_text[0]), 'ents': annotations})
    #visualize_ner(ner_data, out_path='visualized_ner_wikidump.html')

    return wikidump_labeled


def find_token_index(span_position, tokens):
    '''Get token belonging to position in sentence'''
    for i, token in enumerate(tokens):
        if token['start'] <= span_position < token['end']:
            return i
    return -1


def wikidump_to_prodigy(wikidump_labeled, output_path):
    '''Join predictions in span format for separate sentences to
    predictions for the whole article which will load to prodigy.
    '''
    docs = []

    print('CONVERTING SLOVAK BERT WIKI ANNOTATIONS TO PRODIGY FORMAT')
    for doc in tqdm(wikidump_labeled):
        tokens = []
        spans = []
        sentences = ''

        for i, sentence in enumerate(doc['sentences']):

            tokenized_sentence = sentence['text'].split(' ')
            tokens_spans = textspan.get_original_spans(tokenized_sentence, sentence['text'])
            tokens.extend([{
                'id': j + len(tokens),
                'text': text,
                'start': spans[0][0] + len(sentences),
                'end': spans[0][1] + len(sentences),
            } for j, (text, spans) in enumerate(zip(tokenized_sentence, tokens_spans))])

            sentence_spans = []
            for ent in sentence['ents']:

                # fix code after problem with "token_end: -1" (when inference span was with whitespace)
                if ent['text'].strip() != ent['text']:
                    forward_blank_len = len(ent['text'].lstrip()) - len(ent['text'])
                    backward_blank_len = len(ent['text'].rstrip()) - len(ent['text'])
                    ent['text'] = ent['text'].strip()
                    ent['start'] -= forward_blank_len
                    ent['end'] -= backward_blank_len

                sentence_spans.append({
                    'text': ent['text'].strip(),
                    'label': ent['label'],
                    'start': ent['start'] + len(sentences),
                    'end': ent['end'] + len(sentences),
                    'token_start': find_token_index(ent['start'] + len(sentences), tokens),
                    'token_end': find_token_index(ent['end'] + len(sentences) - 1, tokens),
                })
            spans.extend(sentence_spans)

            sentence_separator = ''
            if i != len(doc['sentences']) - 1:
                sentence_separator = '\n\t'
                tokens.append({
                    'id': len(tokens),
                    'text': sentence_separator,
                    'start': len(sentences) + len(sentence['text']),
                    'end': len(sentences) + len(sentence['text']) + len(sentence_separator),
                })

            sentences += sentence['text'] + sentence_separator

        doc_prodigy = {
            'text': sentences,
            'spans': spans,
            'tokens': tokens,
            'meta': doc['meta'],
        }
        docs.append(doc_prodigy)

    with jsonlines.open(output_path, mode='w') as writer:
        writer.write_all(docs)


def main():
    '''Predict and process NER tags on wikipedia articles
    and convert it to Prodigy format for manual annotation
    '''
    ner = pipeline(task='ner', model='./output/test-ner', aggregation_strategy="simple")

    dataset_path = '../wiki_dump/text_jsonl/20220316-170708.jsonl'
    output_path = '../../prodigy/wikiann_bert/20220316-170708_bert.jsonl'

    annotations = inference_wikidump(dataset_path, ner)
    wikidump_to_prodigy(annotations, output_path)

    #visualize_ner({'text': ' '.join(tokenized_text[0]), 'ents': annotations})


if __name__ == '__main__':
    main()

    # Visualizations of inferenced results or data from prodigy
    # annotations_slovakbert = [{'sentences': [{'text': 'Eswatini , dlhý tvar Eswatinské kráľovstvo ( do roku 2018 [ na Slovensku : 2019 ] : Svazijsko , dlhý tvar Svazijské kráľovstvo ) je štát v Afrike .', 'ents': [{'text': 'Eswatinské kráľovstvo', 'label': 'LOC', 'start': 21, 'end': 42}, {'text': 'Slovensku', 'label': 'LOC', 'start': 63, 'end': 72}, {'text': 'Svazijské kráľovstvo', 'label': 'LOC', 'start': 106, 'end': 126}, {'text': 'Afrike', 'label': 'LOC', 'start': 139, 'end': 145}]},
    #                {'text': 'Hlavné mesto je Mbabane .', 'ents': [{'text': 'Hlavné mesto', 'label': 'LOC', 'start': 0, 'end': 12}, {'text': 'babane', 'label': 'LOC', 'start': 17, 'end': 23}]}],
    #                'meta': {'id': '0002'}
    #                }]
    # annotations_prodigy = {"text": "Colin Flooks , známy ako Cozy Powell ( 29. december 1947 , Cirencester , Anglicko – 5. apríl 1998 ) bol britský rockový bubeník .\n\tPowell začal svoju hudobnú kariéru v skupine The Sorcerers v roku 1965 .\n\tV roku 1971 vstúpil do Jeff Beck Group s ktorou nahral dva albumy , potom hral sólovo a založil skupinu Bedlam .\n\tV roku 1973 sa jeho sólový singel `` Dance With The Devil '' umiestnil na treťom mieste v britskom rebríčku hitov .\n\tV nasledujúcom roku sa Cozy preslávil , keď hral naživo v televízii a urobil rekord ako najrýchlejšie hrajúci bubeník .\n\tÚčinkoval aj v televíznom programe BBC pre deti s názvom Record Breakers .\n\tV roku 1976 sa stal členom skupiny Rainbow Ritchieho Blackmorea .\n\tZ jeho účinkovania v skupine je najpamätnejším 16. august 1980 , kedy sa Rainbow preslávili na historicky prvom koncerte Monsters of Rock v Castle Donington v Anglicku .\n\tNasledovala úspešná LP `` Down to Earth '' z ktorej pochádzajú single `` Since You 've Been Gone '' a `` All Night Long '' .\n\tSo spevákom Grahamom Bonnetom neskôr opustil Rainbow , aby začali práce na novom Bonnetovom projekte Bonnet & the Hooligans .\n\tIch spoločným najvýznamnejším singlom bol `` Night Games '' ( 1981 ) .\n\tPowell vystupoval s mnohými známymi skupinami - Michael Schenker Group od 1981 do 1982 , Whitesnake 19821984 , potom s Keithom Emersonom a Gregom Lakeom v roku 1986 a nakoniec so skupinou Black Sabbath v rokoch 1989 - 1991 a 1993 - 1995 .\n\tSpolu s Neilom Murrayom ( bývalým členom Cozy Powell ` s Hammer , Whitesnake a Black Sabbath ) sa stal členom Brian May Bandu a účinkoval na albumoch `` Back To The Light '' a `` Another World '' .\n\tNa jeseň 1998 mali naplánované spoločné turné .\n\tCozy Powell zomrel 5. mája 1998 , na následky dopravnej nehody na diaľnici M4 pri Bristole , keď viedol svoje auto Saab 9000 v nepriaznivom počasí .\n\tPred nehodou nahrával v štúdiu so skupinou Fleetwood Mac .\n\tÚčinkoval ako bubeník na najmenej 66 albumoch .\n\tMal veľký vplyv na hru mnohých rockových bubeníkov a jeho predčasný odchod je veľkou stratou pre rockovú hudbu .\n\tV októbri 2005 sa objavil na trhu nový album s Cozy Powellom .\n\tTony Martin , bývalý spevák Black Sabbath , vydal štúdiový album `` Scream '' , na ktorom použil hudobnú nahrávku s Powellom , ktorá pochádzala z roku 1992 .\n\tObaja hudobníci boli vtedy v skupine Hammer a Cozy mu ju dal na `` ďalšie použitie '' .\n\tExistuje ešte ďalších devätnásť Powellových inštrumentálnych nahrávok , ktoré môžu byť v budúcnosti použité .", "spans": [{"text": "Colin Flooks", "label": "PER", "start": 0, "end": 12, "token_start": 0, "token_end": 1}, {"text": "Cozy Powell", "label": "PER", "start": 25, "end": 36, "token_start": 5, "token_end": 6}, {"text": "Cirencester", "label": "LOC", "start": 59, "end": 70, "token_start": 12, "token_end": 12}, {"text": "Anglicko", "label": "LOC", "start": 73, "end": 81, "token_start": 14, "token_end": 14}, {"text": "The Sorcerers", "label": "ORG", "start": 176, "end": 189, "token_start": 33, "token_end": 34}, {"text": "Jeff Beck Group", "label": "ORG", "start": 228, "end": 243, "token_start": 45, "token_end": 47}, {"text": "Bedlam", "label": "ORG", "start": 309, "end": 315, "token_start": 60, "token_end": 60}, {"text": "Cozy", "label": "PER", "start": 459, "end": 463, "token_start": 90, "token_end": 90}, {"text": "BBC", "label": "ORG", "start": 592, "end": 595, "token_start": 112, "token_end": 112}, {"text": "Record Breakers", "label": "ORG", "start": 614, "end": 629, "token_start": 117, "token_end": 118}, {"text": "Rainbow", "label": "ORG", "start": 668, "end": 675, "token_start": 128, "token_end": 128}, {"text": "Ritchieho Blackmorea", "label": "PER", "start": 676, "end": 696, "token_start": 129, "token_end": 130}, {"text": "Rainbow", "label": "ORG", "start": 773, "end": 780, "token_start": 146, "token_end": 146}, {"text": "Monsters of Rock", "label": "ORG", "start": 821, "end": 837, "token_start": 152, "token_end": 154}, {"text": "Castle Donington", "label": "ORG", "start": 840, "end": 856, "token_start": 156, "token_end": 157}, {"text": "Anglicku", "label": "LOC", "start": 859, "end": 867, "token_start": 159, "token_end": 159}, {"text": "LP", "label": "ORG", "start": 891, "end": 893, "token_start": 164, "token_end": 164}, {"text": "Grahamom Bonnetom", "label": "PER", "start": 1009, "end": 1026, "token_start": 191, "token_end": 192}, {"text": "Rainbow", "label": "ORG", "start": 1042, "end": 1049, "token_start": 195, "token_end": 195}, {"text": "Bonnet & the Hooligans", "label": "ORG", "start": 1098, "end": 1120, "token_start": 204, "token_end": 207}, {"text": "Michael Schenker Group", "label": "PER", "start": 1244, "end": 1266, "token_start": 231, "token_end": 233}, {"text": "Keithom Emersonom", "label": "PER", "start": 1315, "end": 1332, "token_start": 244, "token_end": 245}, {"text": "Gregom Lakeom", "label": "PER", "start": 1335, "end": 1348, "token_start": 247, "token_end": 248}, {"text": "Black Sabbath", "label": "ORG", "start": 1384, "end": 1397, "token_start": 256, "token_end": 257}, {"text": "Neilom Murrayom", "label": "PER", "start": 1444, "end": 1459, "token_start": 271, "token_end": 272}, {"text": "Cozy Powell", "label": "PER", "start": 1477, "end": 1488, "token_start": 276, "token_end": 277}, {"text": "Hammer", "label": "ORG", "start": 1493, "end": 1499, "token_start": 280, "token_end": 280}, {"text": "Black Sabbath", "label": "ORG", "start": 1515, "end": 1528, "token_start": 284, "token_end": 285}, {"text": "Brian May Bandu", "label": "ORG", "start": 1546, "end": 1561, "token_start": 290, "token_end": 292}, {"text": "Back To The Light", "label": "ORG", "start": 1589, "end": 1606, "token_start": 298, "token_end": 301}, {"text": "Another World ", "label": "ORG", "start": 1615, "end": 1629, "token_start": 305, "token_end": -1}, {"text": "Cozy Powell", "label": "PER", "start": 1684, "end": 1695, "token_start": 319, "token_end": 320}, {"text": "Bristole", "label": "LOC", "start": 1766, "end": 1774, "token_start": 334, "token_end": 334}, {"text": "Fleetwood Mac", "label": "ORG", "start": 1877, "end": 1890, "token_start": 354, "token_end": 355}, {"text": "Cozy Powellom", "label": "PER", "start": 2104, "end": 2117, "token_start": 397, "token_end": 398}, {"text": "Tony Martin", "label": "PER", "start": 2121, "end": 2132, "token_start": 401, "token_end": 402}, {"text": "Black Sabbath", "label": "ORG", "start": 2149, "end": 2162, "token_start": 406, "token_end": 407}, {"text": "Hammer", "label": "ORG", "start": 2317, "end": 2323, "token_start": 437, "token_end": 437}, {"text": "Cozy", "label": "PER", "start": 2326, "end": 2330, "token_start": 439, "token_end": 439}, {"text": "Powel", "label": "PER", "start": 2401, "end": 2406, "token_start": 454, "token_end": 454}], "tokens": [{"id": 0, "text": "Colin", "start": 0, "end": 5}, {"id": 1, "text": "Flooks", "start": 6, "end": 12}, {"id": 2, "text": ",", "start": 13, "end": 14}, {"id": 3, "text": "známy", "start": 15, "end": 20}, {"id": 4, "text": "ako", "start": 21, "end": 24}, {"id": 5, "text": "Cozy", "start": 25, "end": 29}, {"id": 6, "text": "Powell", "start": 30, "end": 36}, {"id": 7, "text": "(", "start": 37, "end": 38}, {"id": 8, "text": "29.", "start": 39, "end": 42}, {"id": 9, "text": "december", "start": 43, "end": 51}, {"id": 10, "text": "1947", "start": 52, "end": 56}, {"id": 11, "text": ",", "start": 57, "end": 58}, {"id": 12, "text": "Cirencester", "start": 59, "end": 70}, {"id": 13, "text": ",", "start": 71, "end": 72}, {"id": 14, "text": "Anglicko", "start": 73, "end": 81}, {"id": 15, "text": "–", "start": 82, "end": 83}, {"id": 16, "text": "5.", "start": 84, "end": 86}, {"id": 17, "text": "apríl", "start": 87, "end": 92}, {"id": 18, "text": "1998", "start": 93, "end": 97}, {"id": 19, "text": ")", "start": 98, "end": 99}, {"id": 20, "text": "bol", "start": 100, "end": 103}, {"id": 21, "text": "britský", "start": 104, "end": 111}, {"id": 22, "text": "rockový", "start": 112, "end": 119}, {"id": 23, "text": "bubeník", "start": 120, "end": 127}, {"id": 24, "text": ".", "start": 128, "end": 129}, {"id": 25, "text": "\n\t", "start": 0, "end": 2}, {"id": 26, "text": "Powell", "start": 131, "end": 137}, {"id": 27, "text": "začal", "start": 138, "end": 143}, {"id": 28, "text": "svoju", "start": 144, "end": 149}, {"id": 29, "text": "hudobnú", "start": 150, "end": 157}, {"id": 30, "text": "kariéru", "start": 158, "end": 165}, {"id": 31, "text": "v", "start": 166, "end": 167}, {"id": 32, "text": "skupine", "start": 168, "end": 175}, {"id": 33, "text": "The", "start": 176, "end": 179}, {"id": 34, "text": "Sorcerers", "start": 180, "end": 189}, {"id": 35, "text": "v", "start": 190, "end": 191}, {"id": 36, "text": "roku", "start": 192, "end": 196}, {"id": 37, "text": "1965", "start": 197, "end": 201}, {"id": 38, "text": ".", "start": 202, "end": 203}, {"id": 39, "text": "\n\t", "start": 131, "end": 133}, {"id": 40, "text": "V", "start": 205, "end": 206}, {"id": 41, "text": "roku", "start": 207, "end": 211}, {"id": 42, "text": "1971", "start": 212, "end": 216}, {"id": 43, "text": "vstúpil", "start": 217, "end": 224}, {"id": 44, "text": "do", "start": 225, "end": 227}, {"id": 45, "text": "Jeff", "start": 228, "end": 232}, {"id": 46, "text": "Beck", "start": 233, "end": 237}, {"id": 47, "text": "Group", "start": 238, "end": 243}, {"id": 48, "text": "s", "start": 244, "end": 245}, {"id": 49, "text": "ktorou", "start": 246, "end": 252}, {"id": 50, "text": "nahral", "start": 253, "end": 259}, {"id": 51, "text": "dva", "start": 260, "end": 263}, {"id": 52, "text": "albumy", "start": 264, "end": 270}, {"id": 53, "text": ",", "start": 271, "end": 272}, {"id": 54, "text": "potom", "start": 273, "end": 278}, {"id": 55, "text": "hral", "start": 279, "end": 283}, {"id": 56, "text": "sólovo", "start": 284, "end": 290}, {"id": 57, "text": "a", "start": 291, "end": 292}, {"id": 58, "text": "založil", "start": 293, "end": 300}, {"id": 59, "text": "skupinu", "start": 301, "end": 308}, {"id": 60, "text": "Bedlam", "start": 309, "end": 315}, {"id": 61, "text": ".", "start": 316, "end": 317}, {"id": 62, "text": "\n\t", "start": 205, "end": 207}, {"id": 63, "text": "V", "start": 319, "end": 320}, {"id": 64, "text": "roku", "start": 321, "end": 325}, {"id": 65, "text": "1973", "start": 326, "end": 330}, {"id": 66, "text": "sa", "start": 331, "end": 333}, {"id": 67, "text": "jeho", "start": 334, "end": 338}, {"id": 68, "text": "sólový", "start": 339, "end": 345}, {"id": 69, "text": "singel", "start": 346, "end": 352}, {"id": 70, "text": "``", "start": 353, "end": 355}, {"id": 71, "text": "Dance", "start": 356, "end": 361}, {"id": 72, "text": "With", "start": 362, "end": 366}, {"id": 73, "text": "The", "start": 367, "end": 370}, {"id": 74, "text": "Devil", "start": 371, "end": 376}, {"id": 75, "text": "''", "start": 377, "end": 379}, {"id": 76, "text": "umiestnil", "start": 380, "end": 389}, {"id": 77, "text": "na", "start": 390, "end": 392}, {"id": 78, "text": "treťom", "start": 393, "end": 399}, {"id": 79, "text": "mieste", "start": 400, "end": 406}, {"id": 80, "text": "v", "start": 407, "end": 408}, {"id": 81, "text": "britskom", "start": 409, "end": 417}, {"id": 82, "text": "rebríčku", "start": 418, "end": 426}, {"id": 83, "text": "hitov", "start": 427, "end": 432}, {"id": 84, "text": ".", "start": 433, "end": 434}, {"id": 85, "text": "\n\t", "start": 319, "end": 321}, {"id": 86, "text": "V", "start": 436, "end": 437}, {"id": 87, "text": "nasledujúcom", "start": 438, "end": 450}, {"id": 88, "text": "roku", "start": 451, "end": 455}, {"id": 89, "text": "sa", "start": 456, "end": 458}, {"id": 90, "text": "Cozy", "start": 459, "end": 463}, {"id": 91, "text": "preslávil", "start": 464, "end": 473}, {"id": 92, "text": ",", "start": 474, "end": 475}, {"id": 93, "text": "keď", "start": 476, "end": 479}, {"id": 94, "text": "hral", "start": 480, "end": 484}, {"id": 95, "text": "naživo", "start": 485, "end": 491}, {"id": 96, "text": "v", "start": 492, "end": 493}, {"id": 97, "text": "televízii", "start": 494, "end": 503}, {"id": 98, "text": "a", "start": 504, "end": 505}, {"id": 99, "text": "urobil", "start": 506, "end": 512}, {"id": 100, "text": "rekord", "start": 513, "end": 519}, {"id": 101, "text": "ako", "start": 520, "end": 523}, {"id": 102, "text": "najrýchlejšie", "start": 524, "end": 537}, {"id": 103, "text": "hrajúci", "start": 538, "end": 545}, {"id": 104, "text": "bubeník", "start": 546, "end": 553}, {"id": 105, "text": ".", "start": 554, "end": 555}, {"id": 106, "text": "\n\t", "start": 436, "end": 438}, {"id": 107, "text": "Účinkoval", "start": 557, "end": 566}, {"id": 108, "text": "aj", "start": 567, "end": 569}, {"id": 109, "text": "v", "start": 570, "end": 571}, {"id": 110, "text": "televíznom", "start": 572, "end": 582}, {"id": 111, "text": "programe", "start": 583, "end": 591}, {"id": 112, "text": "BBC", "start": 592, "end": 595}, {"id": 113, "text": "pre", "start": 596, "end": 599}, {"id": 114, "text": "deti", "start": 600, "end": 604}, {"id": 115, "text": "s", "start": 605, "end": 606}, {"id": 116, "text": "názvom", "start": 607, "end": 613}, {"id": 117, "text": "Record", "start": 614, "end": 620}, {"id": 118, "text": "Breakers", "start": 621, "end": 629}, {"id": 119, "text": ".", "start": 630, "end": 631}, {"id": 120, "text": "\n\t", "start": 557, "end": 559}, {"id": 121, "text": "V", "start": 633, "end": 634}, {"id": 122, "text": "roku", "start": 635, "end": 639}, {"id": 123, "text": "1976", "start": 640, "end": 644}, {"id": 124, "text": "sa", "start": 645, "end": 647}, {"id": 125, "text": "stal", "start": 648, "end": 652}, {"id": 126, "text": "členom", "start": 653, "end": 659}, {"id": 127, "text": "skupiny", "start": 660, "end": 667}, {"id": 128, "text": "Rainbow", "start": 668, "end": 675}, {"id": 129, "text": "Ritchieho", "start": 676, "end": 685}, {"id": 130, "text": "Blackmorea", "start": 686, "end": 696}, {"id": 131, "text": ".", "start": 697, "end": 698}, {"id": 132, "text": "\n\t", "start": 633, "end": 635}, {"id": 133, "text": "Z", "start": 700, "end": 701}, {"id": 134, "text": "jeho", "start": 702, "end": 706}, {"id": 135, "text": "účinkovania", "start": 707, "end": 718}, {"id": 136, "text": "v", "start": 719, "end": 720}, {"id": 137, "text": "skupine", "start": 721, "end": 728}, {"id": 138, "text": "je", "start": 729, "end": 731}, {"id": 139, "text": "najpamätnejším", "start": 732, "end": 746}, {"id": 140, "text": "16.", "start": 747, "end": 750}, {"id": 141, "text": "august", "start": 751, "end": 757}, {"id": 142, "text": "1980", "start": 758, "end": 762}, {"id": 143, "text": ",", "start": 763, "end": 764}, {"id": 144, "text": "kedy", "start": 765, "end": 769}, {"id": 145, "text": "sa", "start": 770, "end": 772}, {"id": 146, "text": "Rainbow", "start": 773, "end": 780}, {"id": 147, "text": "preslávili", "start": 781, "end": 791}, {"id": 148, "text": "na", "start": 792, "end": 794}, {"id": 149, "text": "historicky", "start": 795, "end": 805}, {"id": 150, "text": "prvom", "start": 806, "end": 811}, {"id": 151, "text": "koncerte", "start": 812, "end": 820}, {"id": 152, "text": "Monsters", "start": 821, "end": 829}, {"id": 153, "text": "of", "start": 830, "end": 832}, {"id": 154, "text": "Rock", "start": 833, "end": 837}, {"id": 155, "text": "v", "start": 838, "end": 839}, {"id": 156, "text": "Castle", "start": 840, "end": 846}, {"id": 157, "text": "Donington", "start": 847, "end": 856}, {"id": 158, "text": "v", "start": 857, "end": 858}, {"id": 159, "text": "Anglicku", "start": 859, "end": 867}, {"id": 160, "text": ".", "start": 868, "end": 869}, {"id": 161, "text": "\n\t", "start": 700, "end": 702}, {"id": 162, "text": "Nasledovala", "start": 871, "end": 882}, {"id": 163, "text": "úspešná", "start": 883, "end": 890}, {"id": 164, "text": "LP", "start": 891, "end": 893}, {"id": 165, "text": "``", "start": 894, "end": 896}, {"id": 166, "text": "Down", "start": 897, "end": 901}, {"id": 167, "text": "to", "start": 902, "end": 904}, {"id": 168, "text": "Earth", "start": 905, "end": 910}, {"id": 169, "text": "''", "start": 911, "end": 913}, {"id": 170, "text": "z", "start": 914, "end": 915}, {"id": 171, "text": "ktorej", "start": 916, "end": 922}, {"id": 172, "text": "pochádzajú", "start": 923, "end": 933}, {"id": 173, "text": "single", "start": 934, "end": 940}, {"id": 174, "text": "``", "start": 941, "end": 943}, {"id": 175, "text": "Since", "start": 944, "end": 949}, {"id": 176, "text": "You", "start": 950, "end": 953}, {"id": 177, "text": "'ve", "start": 954, "end": 957}, {"id": 178, "text": "Been", "start": 958, "end": 962}, {"id": 179, "text": "Gone", "start": 963, "end": 967}, {"id": 180, "text": "''", "start": 968, "end": 970}, {"id": 181, "text": "a", "start": 971, "end": 972}, {"id": 182, "text": "``", "start": 973, "end": 975}, {"id": 183, "text": "All", "start": 976, "end": 979}, {"id": 184, "text": "Night", "start": 980, "end": 985}, {"id": 185, "text": "Long", "start": 986, "end": 990}, {"id": 186, "text": "''", "start": 991, "end": 993}, {"id": 187, "text": ".", "start": 994, "end": 995}, {"id": 188, "text": "\n\t", "start": 871, "end": 873}, {"id": 189, "text": "So", "start": 997, "end": 999}, {"id": 190, "text": "spevákom", "start": 1000, "end": 1008}, {"id": 191, "text": "Grahamom", "start": 1009, "end": 1017}, {"id": 192, "text": "Bonnetom", "start": 1018, "end": 1026}, {"id": 193, "text": "neskôr", "start": 1027, "end": 1033}, {"id": 194, "text": "opustil", "start": 1034, "end": 1041}, {"id": 195, "text": "Rainbow", "start": 1042, "end": 1049}, {"id": 196, "text": ",", "start": 1050, "end": 1051}, {"id": 197, "text": "aby", "start": 1052, "end": 1055}, {"id": 198, "text": "začali", "start": 1056, "end": 1062}, {"id": 199, "text": "práce", "start": 1063, "end": 1068}, {"id": 200, "text": "na", "start": 1069, "end": 1071}, {"id": 201, "text": "novom", "start": 1072, "end": 1077}, {"id": 202, "text": "Bonnetovom", "start": 1078, "end": 1088}, {"id": 203, "text": "projekte", "start": 1089, "end": 1097}, {"id": 204, "text": "Bonnet", "start": 1098, "end": 1104}, {"id": 205, "text": "&", "start": 1105, "end": 1106}, {"id": 206, "text": "the", "start": 1107, "end": 1110}, {"id": 207, "text": "Hooligans", "start": 1111, "end": 1120}, {"id": 208, "text": ".", "start": 1121, "end": 1122}, {"id": 209, "text": "\n\t", "start": 997, "end": 999}, {"id": 210, "text": "Ich", "start": 1124, "end": 1127}, {"id": 211, "text": "spoločným", "start": 1128, "end": 1137}, {"id": 212, "text": "najvýznamnejším", "start": 1138, "end": 1153}, {"id": 213, "text": "singlom", "start": 1154, "end": 1161}, {"id": 214, "text": "bol", "start": 1162, "end": 1165}, {"id": 215, "text": "``", "start": 1166, "end": 1168}, {"id": 216, "text": "Night", "start": 1169, "end": 1174}, {"id": 217, "text": "Games", "start": 1175, "end": 1180}, {"id": 218, "text": "''", "start": 1181, "end": 1183}, {"id": 219, "text": "(", "start": 1184, "end": 1185}, {"id": 220, "text": "1981", "start": 1186, "end": 1190}, {"id": 221, "text": ")", "start": 1191, "end": 1192}, {"id": 222, "text": ".", "start": 1193, "end": 1194}, {"id": 223, "text": "\n\t", "start": 1124, "end": 1126}, {"id": 224, "text": "Powell", "start": 1196, "end": 1202}, {"id": 225, "text": "vystupoval", "start": 1203, "end": 1213}, {"id": 226, "text": "s", "start": 1214, "end": 1215}, {"id": 227, "text": "mnohými", "start": 1216, "end": 1223}, {"id": 228, "text": "známymi", "start": 1224, "end": 1231}, {"id": 229, "text": "skupinami", "start": 1232, "end": 1241}, {"id": 230, "text": "-", "start": 1242, "end": 1243}, {"id": 231, "text": "Michael", "start": 1244, "end": 1251}, {"id": 232, "text": "Schenker", "start": 1252, "end": 1260}, {"id": 233, "text": "Group", "start": 1261, "end": 1266}, {"id": 234, "text": "od", "start": 1267, "end": 1269}, {"id": 235, "text": "1981", "start": 1270, "end": 1274}, {"id": 236, "text": "do", "start": 1275, "end": 1277}, {"id": 237, "text": "1982", "start": 1278, "end": 1282}, {"id": 238, "text": ",", "start": 1283, "end": 1284}, {"id": 239, "text": "Whitesnake", "start": 1285, "end": 1295}, {"id": 240, "text": "19821984", "start": 1296, "end": 1304}, {"id": 241, "text": ",", "start": 1305, "end": 1306}, {"id": 242, "text": "potom", "start": 1307, "end": 1312}, {"id": 243, "text": "s", "start": 1313, "end": 1314}, {"id": 244, "text": "Keithom", "start": 1315, "end": 1322}, {"id": 245, "text": "Emersonom", "start": 1323, "end": 1332}, {"id": 246, "text": "a", "start": 1333, "end": 1334}, {"id": 247, "text": "Gregom", "start": 1335, "end": 1341}, {"id": 248, "text": "Lakeom", "start": 1342, "end": 1348}, {"id": 249, "text": "v", "start": 1349, "end": 1350}, {"id": 250, "text": "roku", "start": 1351, "end": 1355}, {"id": 251, "text": "1986", "start": 1356, "end": 1360}, {"id": 252, "text": "a", "start": 1361, "end": 1362}, {"id": 253, "text": "nakoniec", "start": 1363, "end": 1371}, {"id": 254, "text": "so", "start": 1372, "end": 1374}, {"id": 255, "text": "skupinou", "start": 1375, "end": 1383}, {"id": 256, "text": "Black", "start": 1384, "end": 1389}, {"id": 257, "text": "Sabbath", "start": 1390, "end": 1397}, {"id": 258, "text": "v", "start": 1398, "end": 1399}, {"id": 259, "text": "rokoch", "start": 1400, "end": 1406}, {"id": 260, "text": "1989", "start": 1407, "end": 1411}, {"id": 261, "text": "-", "start": 1412, "end": 1413}, {"id": 262, "text": "1991", "start": 1414, "end": 1418}, {"id": 263, "text": "a", "start": 1419, "end": 1420}, {"id": 264, "text": "1993", "start": 1421, "end": 1425}, {"id": 265, "text": "-", "start": 1426, "end": 1427}, {"id": 266, "text": "1995", "start": 1428, "end": 1432}, {"id": 267, "text": ".", "start": 1433, "end": 1434}, {"id": 268, "text": "\n\t", "start": 1196, "end": 1198}, {"id": 269, "text": "Spolu", "start": 1436, "end": 1441}, {"id": 270, "text": "s", "start": 1442, "end": 1443}, {"id": 271, "text": "Neilom", "start": 1444, "end": 1450}, {"id": 272, "text": "Murrayom", "start": 1451, "end": 1459}, {"id": 273, "text": "(", "start": 1460, "end": 1461}, {"id": 274, "text": "bývalým", "start": 1462, "end": 1469}, {"id": 275, "text": "členom", "start": 1470, "end": 1476}, {"id": 276, "text": "Cozy", "start": 1477, "end": 1481}, {"id": 277, "text": "Powell", "start": 1482, "end": 1488}, {"id": 278, "text": "`", "start": 1489, "end": 1490}, {"id": 279, "text": "s", "start": 1491, "end": 1492}, {"id": 280, "text": "Hammer", "start": 1493, "end": 1499}, {"id": 281, "text": ",", "start": 1500, "end": 1501}, {"id": 282, "text": "Whitesnake", "start": 1502, "end": 1512}, {"id": 283, "text": "a", "start": 1513, "end": 1514}, {"id": 284, "text": "Black", "start": 1515, "end": 1520}, {"id": 285, "text": "Sabbath", "start": 1521, "end": 1528}, {"id": 286, "text": ")", "start": 1529, "end": 1530}, {"id": 287, "text": "sa", "start": 1531, "end": 1533}, {"id": 288, "text": "stal", "start": 1534, "end": 1538}, {"id": 289, "text": "členom", "start": 1539, "end": 1545}, {"id": 290, "text": "Brian", "start": 1546, "end": 1551}, {"id": 291, "text": "May", "start": 1552, "end": 1555}, {"id": 292, "text": "Bandu", "start": 1556, "end": 1561}, {"id": 293, "text": "a", "start": 1562, "end": 1563}, {"id": 294, "text": "účinkoval", "start": 1564, "end": 1573}, {"id": 295, "text": "na", "start": 1574, "end": 1576}, {"id": 296, "text": "albumoch", "start": 1577, "end": 1585}, {"id": 297, "text": "``", "start": 1586, "end": 1588}, {"id": 298, "text": "Back", "start": 1589, "end": 1593}, {"id": 299, "text": "To", "start": 1594, "end": 1596}, {"id": 300, "text": "The", "start": 1597, "end": 1600}, {"id": 301, "text": "Light", "start": 1601, "end": 1606}, {"id": 302, "text": "''", "start": 1607, "end": 1609}, {"id": 303, "text": "a", "start": 1610, "end": 1611}, {"id": 304, "text": "``", "start": 1612, "end": 1614}, {"id": 305, "text": "Another", "start": 1615, "end": 1622}, {"id": 306, "text": "World", "start": 1623, "end": 1628}, {"id": 307, "text": "''", "start": 1629, "end": 1631}, {"id": 308, "text": ".", "start": 1632, "end": 1633}, {"id": 309, "text": "\n\t", "start": 1436, "end": 1438}, {"id": 310, "text": "Na", "start": 1635, "end": 1637}, {"id": 311, "text": "jeseň", "start": 1638, "end": 1643}, {"id": 312, "text": "1998", "start": 1644, "end": 1648}, {"id": 313, "text": "mali", "start": 1649, "end": 1653}, {"id": 314, "text": "naplánované", "start": 1654, "end": 1665}, {"id": 315, "text": "spoločné", "start": 1666, "end": 1674}, {"id": 316, "text": "turné", "start": 1675, "end": 1680}, {"id": 317, "text": ".", "start": 1681, "end": 1682}, {"id": 318, "text": "\n\t", "start": 1635, "end": 1637}, {"id": 319, "text": "Cozy", "start": 1684, "end": 1688}, {"id": 320, "text": "Powell", "start": 1689, "end": 1695}, {"id": 321, "text": "zomrel", "start": 1696, "end": 1702}, {"id": 322, "text": "5.", "start": 1703, "end": 1705}, {"id": 323, "text": "mája", "start": 1706, "end": 1710}, {"id": 324, "text": "1998", "start": 1711, "end": 1715}, {"id": 325, "text": ",", "start": 1716, "end": 1717}, {"id": 326, "text": "na", "start": 1718, "end": 1720}, {"id": 327, "text": "následky", "start": 1721, "end": 1729}, {"id": 328, "text": "dopravnej", "start": 1730, "end": 1739}, {"id": 329, "text": "nehody", "start": 1740, "end": 1746}, {"id": 330, "text": "na", "start": 1747, "end": 1749}, {"id": 331, "text": "diaľnici", "start": 1750, "end": 1758}, {"id": 332, "text": "M4", "start": 1759, "end": 1761}, {"id": 333, "text": "pri", "start": 1762, "end": 1765}, {"id": 334, "text": "Bristole", "start": 1766, "end": 1774}, {"id": 335, "text": ",", "start": 1775, "end": 1776}, {"id": 336, "text": "keď", "start": 1777, "end": 1780}, {"id": 337, "text": "viedol", "start": 1781, "end": 1787}, {"id": 338, "text": "svoje", "start": 1788, "end": 1793}, {"id": 339, "text": "auto", "start": 1794, "end": 1798}, {"id": 340, "text": "Saab", "start": 1799, "end": 1803}, {"id": 341, "text": "9000", "start": 1804, "end": 1808}, {"id": 342, "text": "v", "start": 1809, "end": 1810}, {"id": 343, "text": "nepriaznivom", "start": 1811, "end": 1823}, {"id": 344, "text": "počasí", "start": 1824, "end": 1830}, {"id": 345, "text": ".", "start": 1831, "end": 1832}, {"id": 346, "text": "\n\t", "start": 1684, "end": 1686}, {"id": 347, "text": "Pred", "start": 1834, "end": 1838}, {"id": 348, "text": "nehodou", "start": 1839, "end": 1846}, {"id": 349, "text": "nahrával", "start": 1847, "end": 1855}, {"id": 350, "text": "v", "start": 1856, "end": 1857}, {"id": 351, "text": "štúdiu", "start": 1858, "end": 1864}, {"id": 352, "text": "so", "start": 1865, "end": 1867}, {"id": 353, "text": "skupinou", "start": 1868, "end": 1876}, {"id": 354, "text": "Fleetwood", "start": 1877, "end": 1886}, {"id": 355, "text": "Mac", "start": 1887, "end": 1890}, {"id": 356, "text": ".", "start": 1891, "end": 1892}, {"id": 357, "text": "\n\t", "start": 1834, "end": 1836}, {"id": 358, "text": "Účinkoval", "start": 1894, "end": 1903}, {"id": 359, "text": "ako", "start": 1904, "end": 1907}, {"id": 360, "text": "bubeník", "start": 1908, "end": 1915}, {"id": 361, "text": "na", "start": 1916, "end": 1918}, {"id": 362, "text": "najmenej", "start": 1919, "end": 1927}, {"id": 363, "text": "66", "start": 1928, "end": 1930}, {"id": 364, "text": "albumoch", "start": 1931, "end": 1939}, {"id": 365, "text": ".", "start": 1940, "end": 1941}, {"id": 366, "text": "\n\t", "start": 1894, "end": 1896}, {"id": 367, "text": "Mal", "start": 1943, "end": 1946}, {"id": 368, "text": "veľký", "start": 1947, "end": 1952}, {"id": 369, "text": "vplyv", "start": 1953, "end": 1958}, {"id": 370, "text": "na", "start": 1959, "end": 1961}, {"id": 371, "text": "hru", "start": 1962, "end": 1965}, {"id": 372, "text": "mnohých", "start": 1966, "end": 1973}, {"id": 373, "text": "rockových", "start": 1974, "end": 1983}, {"id": 374, "text": "bubeníkov", "start": 1984, "end": 1993}, {"id": 375, "text": "a", "start": 1994, "end": 1995}, {"id": 376, "text": "jeho", "start": 1996, "end": 2000}, {"id": 377, "text": "predčasný", "start": 2001, "end": 2010}, {"id": 378, "text": "odchod", "start": 2011, "end": 2017}, {"id": 379, "text": "je", "start": 2018, "end": 2020}, {"id": 380, "text": "veľkou", "start": 2021, "end": 2027}, {"id": 381, "text": "stratou", "start": 2028, "end": 2035}, {"id": 382, "text": "pre", "start": 2036, "end": 2039}, {"id": 383, "text": "rockovú", "start": 2040, "end": 2047}, {"id": 384, "text": "hudbu", "start": 2048, "end": 2053}, {"id": 385, "text": ".", "start": 2054, "end": 2055}, {"id": 386, "text": "\n\t", "start": 1943, "end": 1945}, {"id": 387, "text": "V", "start": 2057, "end": 2058}, {"id": 388, "text": "októbri", "start": 2059, "end": 2066}, {"id": 389, "text": "2005", "start": 2067, "end": 2071}, {"id": 390, "text": "sa", "start": 2072, "end": 2074}, {"id": 391, "text": "objavil", "start": 2075, "end": 2082}, {"id": 392, "text": "na", "start": 2083, "end": 2085}, {"id": 393, "text": "trhu", "start": 2086, "end": 2090}, {"id": 394, "text": "nový", "start": 2091, "end": 2095}, {"id": 395, "text": "album", "start": 2096, "end": 2101}, {"id": 396, "text": "s", "start": 2102, "end": 2103}, {"id": 397, "text": "Cozy", "start": 2104, "end": 2108}, {"id": 398, "text": "Powellom", "start": 2109, "end": 2117}, {"id": 399, "text": ".", "start": 2118, "end": 2119}, {"id": 400, "text": "\n\t", "start": 2057, "end": 2059}, {"id": 401, "text": "Tony", "start": 2121, "end": 2125}, {"id": 402, "text": "Martin", "start": 2126, "end": 2132}, {"id": 403, "text": ",", "start": 2133, "end": 2134}, {"id": 404, "text": "bývalý", "start": 2135, "end": 2141}, {"id": 405, "text": "spevák", "start": 2142, "end": 2148}, {"id": 406, "text": "Black", "start": 2149, "end": 2154}, {"id": 407, "text": "Sabbath", "start": 2155, "end": 2162}, {"id": 408, "text": ",", "start": 2163, "end": 2164}, {"id": 409, "text": "vydal", "start": 2165, "end": 2170}, {"id": 410, "text": "štúdiový", "start": 2171, "end": 2179}, {"id": 411, "text": "album", "start": 2180, "end": 2185}, {"id": 412, "text": "``", "start": 2186, "end": 2188}, {"id": 413, "text": "Scream", "start": 2189, "end": 2195}, {"id": 414, "text": "''", "start": 2196, "end": 2198}, {"id": 415, "text": ",", "start": 2199, "end": 2200}, {"id": 416, "text": "na", "start": 2201, "end": 2203}, {"id": 417, "text": "ktorom", "start": 2204, "end": 2210}, {"id": 418, "text": "použil", "start": 2211, "end": 2217}, {"id": 419, "text": "hudobnú", "start": 2218, "end": 2225}, {"id": 420, "text": "nahrávku", "start": 2226, "end": 2234}, {"id": 421, "text": "s", "start": 2235, "end": 2236}, {"id": 422, "text": "Powellom", "start": 2237, "end": 2245}, {"id": 423, "text": ",", "start": 2246, "end": 2247}, {"id": 424, "text": "ktorá", "start": 2248, "end": 2253}, {"id": 425, "text": "pochádzala", "start": 2254, "end": 2264}, {"id": 426, "text": "z", "start": 2265, "end": 2266}, {"id": 427, "text": "roku", "start": 2267, "end": 2271}, {"id": 428, "text": "1992", "start": 2272, "end": 2276}, {"id": 429, "text": ".", "start": 2277, "end": 2278}, {"id": 430, "text": "\n\t", "start": 2121, "end": 2123}, {"id": 431, "text": "Obaja", "start": 2280, "end": 2285}, {"id": 432, "text": "hudobníci", "start": 2286, "end": 2295}, {"id": 433, "text": "boli", "start": 2296, "end": 2300}, {"id": 434, "text": "vtedy", "start": 2301, "end": 2306}, {"id": 435, "text": "v", "start": 2307, "end": 2308}, {"id": 436, "text": "skupine", "start": 2309, "end": 2316}, {"id": 437, "text": "Hammer", "start": 2317, "end": 2323}, {"id": 438, "text": "a", "start": 2324, "end": 2325}, {"id": 439, "text": "Cozy", "start": 2326, "end": 2330}, {"id": 440, "text": "mu", "start": 2331, "end": 2333}, {"id": 441, "text": "ju", "start": 2334, "end": 2336}, {"id": 442, "text": "dal", "start": 2337, "end": 2340}, {"id": 443, "text": "na", "start": 2341, "end": 2343}, {"id": 444, "text": "``", "start": 2344, "end": 2346}, {"id": 445, "text": "ďalšie", "start": 2347, "end": 2353}, {"id": 446, "text": "použitie", "start": 2354, "end": 2362}, {"id": 447, "text": "''", "start": 2363, "end": 2365}, {"id": 448, "text": ".", "start": 2366, "end": 2367}, {"id": 449, "text": "\n\t", "start": 2280, "end": 2282}, {"id": 450, "text": "Existuje", "start": 2369, "end": 2377}, {"id": 451, "text": "ešte", "start": 2378, "end": 2382}, {"id": 452, "text": "ďalších", "start": 2383, "end": 2390}, {"id": 453, "text": "devätnásť", "start": 2391, "end": 2400}, {"id": 454, "text": "Powellových", "start": 2401, "end": 2412}, {"id": 455, "text": "inštrumentálnych", "start": 2413, "end": 2429}, {"id": 456, "text": "nahrávok", "start": 2430, "end": 2438}, {"id": 457, "text": ",", "start": 2439, "end": 2440}, {"id": 458, "text": "ktoré", "start": 2441, "end": 2446}, {"id": 459, "text": "môžu", "start": 2447, "end": 2451}, {"id": 460, "text": "byť", "start": 2452, "end": 2455}, {"id": 461, "text": "v", "start": 2456, "end": 2457}, {"id": 462, "text": "budúcnosti", "start": 2458, "end": 2468}, {"id": 463, "text": "použité", "start": 2469, "end": 2476}, {"id": 464, "text": ".", "start": 2477, "end": 2478}], "meta": {"id": "77796"}}
    #visualize_prodigy(annotations_prodigy)