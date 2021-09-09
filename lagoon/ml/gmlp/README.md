# Graph-based Multi-layer perceptron (GMLP)

## Algorithm
```
Given:
A time window
persons = all persons in this time window
targets = targets for all these persons in this time window

for person,target in persons,targets:
    x1 = sum of toxicity features for this person and its 1st hop neighbors within the time window
    x2 = sum of toxicity features for this person and its 1st and 2nd hop neighbors within the time window
    x1 = MLP1(x1)
    x2 = MLP2(x2)
    x = concat(x1,x2)
    x = MLP(x) #single regression output neuron
    Loss = L1(x,target)
```

## Data
`../data/persons_toxicity/`

## Running
Run `run_network_wrapper()`. Variables are self-explanatory.

