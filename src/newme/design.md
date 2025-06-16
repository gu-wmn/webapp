# Design

## Main (browse/search)

### Order alphabetically by label

Short alpha-summary
```(text)
LABEL_NAME: STRING sequences: [LINK]WMN_ID(occurences)[/LINK], [LINK]WMN_ID(occurences)[/LINK]
```

Full alpha-summary?
```(text)
label: LABEL_NAME
trigger word / indicator sentence: STRING
sequences:
    [LINK: WMN_ID]
    wmn: WMN_ID, CORPUS, (CONTEXT)?
    wmn type: WMN, wmn_meaning: WMNMeaning
    excerpt: str
    [/LINK]
    [LINK: WMN_ID]
    wmn: WMN_ID, CORPUS, (CONTEXT)?
    wmn type: WMN, wmn_meaning: WMNMeaning
    excerpt: str
    [/LINK]
...
```

### Order by WMN

Short wmn-summary
```(text)
WMN_ID
    LABEL_NAME: STRING(occurences)
    LABEL_NAME: STRING(occurences)
```

Full wmn-summary
```(text)
wmn id: WMN_ID
wmn type: WMN
wmn_meaning: WMNMeaning
context: CONTEXT
LABEL_NAME: str
LABEL_NAME: str
```

## Search function flow

- sort by:
    - alphabetical:
        - get_summaries(group_by="label")
    - wmn
        - get_summaries(group_by="wmn")

### get_summaries_by_wmn_sequence()

```(python)

```
