### Prodigy usage
  
- simple example
```
  wget https://raw.githubusercontent.com/explosion/prodigy-recipes/master/example-datasets/news_headlines.jsonl
  prodigy ner.manual ner_news_headlines blank:en ./news_headlines.jsonl --label PERSON,ORG,PRODUCT,LOCATION
  # follow printed instructions and open application
  # annotate some date
  prodigy db-out ner_news_headlines > ./annotations.jsonl
```

- my usage
```
prodigy ner.manual prod_<n> blank:en wikidump/<predictions file from slovakbert> --label PER,ORG,LOC,MISC`
prodigy db-out prod_<n> > ./annotations_prod<n>.jsonl
```

### Content of this folder
- `prodigy-data/prodigy.json` - config file for prodigy tool
- `wikiann_bert/` - predictions of SlovakBERT trained on WikiANN on data ready for manual annotation
- `prodigy_jsonl/` - data after manual annotation in prodigy




### My usage
