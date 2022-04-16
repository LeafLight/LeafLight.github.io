---
title: nltk and Grammar -- Encoding Part
date: 2022-04-16 21:14:59
tags: ["nltk", "molecules", "GrammarVAE", "MachineLearning", "CMap"]
categories: MachineLearning
---
## Background

When dealing with the recent project associated with _CMap_, an interesting neural network model called __GVAE__ caught my attention. After learning the details of it, I tried to re-do the model and this blog is a recording in some ways and mainly about the _Grammar_ part.

## Grammar -- Context-Free Grammar(CFG)

The key feature of GVAE is __CFG__, which can be manipulated easily by the python module `nltk`.

1. Generate a `CFG` object from string
```python
import nltk
from nltk import CFG

SMILEsGrammar = CFG.fromstring(
"""
    smiles -> chain
    atom -> bracket_atom | aliphatic_organic | aromatic_organic
    aliphatic_organic -> 'B' | 'C' | 'N' | 'O' | 'S' | 'P' | 'F' | 'I' | 'Cl' | 'Br'
    aromatic_organic -> '[' BAI ']'
    BAI -> isotope symbol BAC | symbol BAC | isotope symbol | symbol
    BAC -> chiral BAH | BAH | chiral
    BAH -> hcount BACH | BACH | hcount
    BACH -> charge class | charge | class
    symbol -> aliphatic_organic | aromatic_organic
    isotope -> DIGIT | DIGIT DIGIT | DIGIT DIGIT DIGIT
    DIGIT -> '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8'
    chiral -> '@' | '@@'
    hcount -> 'H' | 'H' DIGIT
    charge -> '-' | '-' DIGIT | '-' DIGIT DIGIT | '+' | '+' DIGIT | '+' DIGIT DIGIT
    bond -> '-' | '=' | '#' | '/' | '\\'
    ringbond -> DIGIT | bond DIGIT
    branched_atom -> atom | atom RB | atom BB | atom RB BB
    RB -> RB ringbond | ringbond
    BB -> BB branch | branch
    branch -> '(' chain ')' | '(' bond chain ')'
    chain -> branched_atom | chain branched_atom | chain bond branched_atom

"""
)
```

2. Generate a parse tree of a molecule in the form of SMILEs
	1. Get a tokenizer
		Since some leaves in the grammar has more than one charactor(like "Cl" or "Br"), default tokenizer may result in errors.
		```python
def get_zinc_tokenizer(cfg):
    # get all the long tokens for the following replacement work
    # long tokens: tokens with more than one charactor, like 'Br'
    long_tokens = [a for a in list(SMILEsGrammar._lexical_index.keys()) if xlength(a) > 1]
    # char used for replacements of 'Cl', 'Br', '@@'
    replacements = ['$', '%', '^']
    # ensure that we have  paired origin tokens and their replacements
    assert xlength(long_tokens) == len(replacements)
    # ensure that all the tokens for replacement is available: not in the origin dict of grammar
    for token in replacements:
        assert token not in cfg._lexical_index

    # the func to return
    def tokenize(smiles):
        # replace all the long_tokens in the input SMILEs
        for i, token in enumerate(long_tokens):
            smiles = smiles.replace(token, replacements[i])
        # the result variable init
        tokens = []
        for token in smiles:
            try:
                # try to find the replaced elements' index and append the original elements 
                ix = replacements.index(token)
                tokens.append(long_tokens[ix])
            except:
                tokens.append(token)
        return tokens
    return tokenize
		```
	2. tokenize the SMILEs string
	```python
smi = "CC1=CC(=O)C2=C(O1)C=C3C(=C2OC)C=CO3"
SMILEs_tokenizer = get_zinc_tokenizer(SMILEsGrammar)
smi_t = SMILEs_tokenizer(smi)
	```
	3. generate the parse tree of the given SMILEs string
	```python
SMILEs_parser = nltk.ChartParser(SMILEsGrammar)
smi_s = next(SMILEs_parser.parse(smi_t))
type(smi_s)
# nltk.tree.tree.Tree
	```
	4. regenerate the SMILEs from the tree
	```python
''.join(smi_s.leaves()
	```
3. Get the productions-index map dict for one-hot encode
	1. productions of the parse tree
	```python
smi_s.productions()
	```
	2. productions-index map dict
	```python
Prod_map = {}
for ix, prod in enumerate(SMILEsGrammar.productions()):
	Prod_map[prod] = ix
	```
	3. one-hot encoding
	```python
import numpy as np
# a batch of smiles strings as example
smiles = "here is a list of smiles"

smiles_t = map(SMILEs_tokenizer, smiles)

smiles_parse_trees = []
for i, t in enumerate(smiles_t):
	smiles_parse_trees[i] = SMILEs_parser.parse(t)

productions_seq = [tree.productions() for tree in smiles_parse_trees]

indices = [np.array([Prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]

MAX_LEN = 277
n_char = len(SMILEsGrammar.productions())

# init 
one_hot = np.zeros((len(indices), MAX_LEN, n_char),dtype=np.float32)

for i in range(len(indices)):
	num_productions = len(indices[i])
	if num_productions > MAX_LEN:
		print("Too Large molecule, out of range")
		one_hot[i][np.arange(MAX_LEN), indices[i][:MAX_LEN]] = 1
	else:
		one_hot[i][np.arange(num_productions), indices[i]] = 1
		one_hot[i][np.arange(num_productions, MAX_LEN, -1)] = 1
	```

