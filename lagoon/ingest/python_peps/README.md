# Python Enhancement Proposals (PEPs)

## General info on PEPs

- PEP types:
    - Standards Track – New feature.
    - Process – Like Standards Track, but doesn't directly apply to Python.
    - Informational – General guidelines.
- We probably want to focus on Standards Track and Process for ML purposes.
- PEP statuses:

<img style="background-color:white" src="https://s3.dualstack.us-east-2.amazonaws.com/pythondotorg-assets/media/dev/peps/pep-0001/pep-0001-process_flow.png">

- PEPs may connect to separate repos where the code is developed.
- Each PEP has a champion / author.
- Discussion is done in:
    - python-dev@python.org
    - python-ideas@python.org
- Relevant discussion messages can be found in 2 ways:
    - Search the mailing list archives for `PEP xxx`. This returns all messages which have the term `PEP xxx`. Each message in a thread is listed separately, which is great.
    - Sometimes the PEP page has a bunch of references to related messages. ([Example for PEP 435](https://www.python.org/dev/peps/pep-0435/#references)). This is a small set of messages which are significant in the lifetime of the PEP. Some of these messages may not explicitly have the words `PEP xxx` and hence will not show up in the previous search. This is why these should be considered separately. When considering these, make sure to get all messages in that thread.

## Info on code
The code reads the raw HTML data from the [PEPs page on the Python website](https://www.python.org/dev/peps/). It can optionally store the raw HTML data as tables in `./peps.csv` and `./authors.csv` if `get_peps()` in `./peps.py` is run with `write_peps_to_csv = True` and `write_authors_to_csv = True`, respectively. These CSV files not committed to Git.

Note that some data on PEP authors is collected for ML/stats purposes and stored as `authors_stats.csv` in the [`lagoon-artifacts` repo](https://gitlab-ext.galois.com/lagoon/lagoon-artifacts/-/blob/main/results/python_peps/authors_stats.csv). The number of rows there is less than the number of rows in `./authors.csv` by 2 because `./authors.csv` includes the following 2 pairs, each of which is actually the same person:
```
von Löwis,Martin,von Löwis,,martin@v.loewis.de
v. Löwis,Martin,v. Löwis,,martin@v.loewis.de
```
and
```
Galindo,Pablo,Galindo,,pablogsal@python.org
Salgado,Pablo Galindo,Salgado,,pablogsal@python.org
```
Note that `authors_stats.csv` is after entity fusion and so these duplicates are corrected, as evidenced by the PEPs attributed to these authors.
