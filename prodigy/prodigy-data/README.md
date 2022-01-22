### Simple example
  
```
  wget https://raw.githubusercontent.com/explosion/prodigy-recipes/master/example-datasets/news_headlines.jsonl
  prodigy ner.manual ner_news_headlines blank:en ./news_headlines.jsonl --label PERSON,ORG,PRODUCT,LOCATION
  # follow printed instructions and open application
  # annotate some date
  prodigy db-out ner_news_headlines > ./annotations.jsonl
```

### Content of this folder
- `prodigy.json` - config file
- `<timestamp>_bert.jsonl` - annotated texts from wikiann bert model ready for manual labeling
- `annotations_<number>.jsonl` - annotated data from prodigy
- `annotations_<number>.conll` - annotated data from prodigy coverted to conll2003 format