# Annotation manual

## PER
- **titles** are not part of PER entity
- **groups of people** that belongs to nation, city, family..., `Slováci`, `Bratislavčania`, `Kováčovci`
- general **adjectives** are not entities e.g. `rímsky vojak`, `slovensky jazyk`, but personal adjectives are PER entity,
  e.g. `to je Petrov kufor` -> `Petrov` is PER entity 

## ORG
- political parties, companies, institutions, political/sport/educational organizations, music bands, restaurants...
- `ESET, spol. s r.o.` -> this is all one ORG entity

## LOC
- rivers, castles, buildings, parks, bridges, zoos, streets (without numbers)...

## MISC
- names of movies, awards (if it's not award of some organization), newspapers, events


## Tips
- **abbreviations**: 
  - separately from word before, example: `Ministerstvo vnútra (MV)...` -> 2 separate ORG tags
  - if you don't know what abbreviation is - google it
  - some abbr. are not entities - e.g. ZŠ (if it's not in name of school), PVC
- **inserted entity**, example: `Krajský súd Pezinok...`, `Andrej Šeban Band` -> it's all only one ORG entity
- **aposthrophes**, `"Afri" je kmen...` -> only `Afri` without `"` is PER entity
- **divied entities**, `Malé a Velké Leváre...` -> 2 LOC entities: `Malé` and `Velké Leváre`
