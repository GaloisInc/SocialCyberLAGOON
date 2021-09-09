# GraphSage
[Original paper](https://arxiv.org/abs/1706.02216)

## Algorithm
```
Given:
A time window
persons = all persons in this time window
targets = targets for all these persons in this time window

for person,target in persons,targets:
    for node_i in concat(person, this person's 1st hop neighbors within the time window):
        for node_j in node_i's 1st hop neighbors within the time window:
            x_j = toxicity features for node_j
            x_j = AggrMLP1(x_j)
        x_i = max(x_j)
        x_i = EmbedMLP1(x_i)
        x_i = AggrMLP2(x_i)
    x = max(x_i)
    x = EmbedMLP2(x)
    x = MLP(x) #single regression output neuron
    Loss = L1(x,target)
```

## Data
`../data/data_toxicity/`

## Running
Run `run_network_wrapper()`. Variables are self-explanatory.

