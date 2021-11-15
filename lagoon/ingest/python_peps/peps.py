import os
import csv
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any


def get_soup(url: str) -> BeautifulSoup:
    """ Get the BeautifulSoup scrape of a URL """
    return BeautifulSoup(requests.get(url).content, "html.parser")


def get_authors() -> Dict[str,Dict[str,Any]]:
    """
    Return: authors_dict
        Keys: How the authors are referenced in the PEP numerical index, e.g. 'GvR'
        Values: Dict containing info about the author
    """
    pep0_soup = get_soup(url="https://www.python.org/dev/peps/")
    authors_soup = pep0_soup.find(id="authors-owners")
    authors_rows = authors_soup.find("tbody").find_all("tr")
    authors_dict = {}
    for row in authors_rows:
        name, email = row.find_all("td")
        names = name.text.split(', ')
        last_name = names[0]
        first_name = names[1] if len(names)>1 else ''
        suffix = names[2] if len(names)>2 else ''
        email = '' if email.text == '\xa0' else email.text.replace(' at ','@')
        
        # Form the key
        # Normally this is the same as the last name, but some special cases are treated separately
        if first_name == 'Guido (GvR)':
            first_name = 'Guido'
            key = 'GvR'
        elif first_name == 'Just (JvR)':
            first_name = 'Just'
            key = 'JvR'
        else:
            key = last_name
        
        authors_dict[key] = {
            'last_name': last_name,
            'first_name': first_name,
            'suffix': suffix,
            'email': email
        }
    return authors_dict


def get_peps(write_peps_to_csv: bool = False, write_authors_to_csv: bool = False) -> Dict[int,Dict[str,Any]]:
    """
    Return PEPs as a dataframe
    If the write_*_to_csv arguments are set, the data is written to CSVs. Useful for debugging.
    """
    pep0_soup = get_soup(url="https://www.python.org/dev/peps/")
    authors_dict = get_authors()
    num_index_rows = pep0_soup.find(id="numerical-index").find("tbody").find_all("tr")
    peps = {}
    
    for row in tqdm(num_index_rows):
        _, row_number, row_title, row_authors = row.find_all("td")
        # First element is row code, which we will get from the PEP page instead of here.
        # This is because row code status can sometimes be blank, while the PEP page gives the actual status

        ## PEP number
        number = row_number.text

        ## PEP url
        url = f"https://www.python.org{row_number.find('a')['href']}"

        ## PEP title
        title = row_title.text

        ## PEP authors and emails
        authors = '' #this will finally store all full names and emails in "<first name> <last name> <<email>> format"
        keys = row_authors.text.split(', ')
        for key in keys:
            try:
                authors += f"{authors_dict[key]['first_name']} {authors_dict[key]['last_name']} {authors_dict[key]['suffix']} <{authors_dict[key]['email']}>; "
            except KeyError:
                print(f"WARNING: Key {key} not found, PEP {number} will be ignored")
        authors = authors.strip().replace('  ',' ')
        if authors.endswith(';'):
            authors = authors[:-1]

        ## Other PEP attributes obtained from the specific PEP page
        pep_soup = get_soup(url=url)
        info_table_rows = pep_soup.find("table", class_="rfc2822 docutils field-list").find_all("tr")
        type_ = status = created = replaces = superseded_by = requires = '' # in case any of these fields are missing, they will default to ''
        for info_table_row in info_table_rows:
            
            if info_table_row.find("th").text.startswith("Type"):
                type_ = info_table_row.find("td").text #NOTE: possible types = ['Process', 'Informational', 'Standards Track']
            elif info_table_row.find("th").text.startswith("Status"):
                status = info_table_row.find("td").text #NOTE: possible statuses = ['Active', 'Superseded', 'Withdrawn', 'Rejected', 'Final', 'Deferred', 'April Fool!', 'Accepted', 'Draft', 'Provisional']
            
            elif info_table_row.find("th").text.startswith("Created"):
                created = info_table_row.find("td").text
            
            # We should capture both replaces and superseded by because not all PEPs have complete info
            # For example, PEP 3124 'Replaces' PEPs 245 and 246, but those 2 don't have 'Superseded-by 3124' on their pages
            elif info_table_row.find("th").text.startswith("Replaces"):
                replaces = info_table_row.find("td").text.strip()
            elif info_table_row.find("th").text.startswith("Superseded"):
                superseded_by = info_table_row.find("td").text.strip()

            elif info_table_row.find("th").text.startswith("Requires"):
                requires = info_table_row.find("td").text.strip()

        ## Append to list
        peps[int(number)] = {
            'url': url,
            'type': type_,
            'status': status,
            'title': title,
            'authors': authors,
            'created': created,
            'replaces': replaces,
            'superseded_by': superseded_by,
            'requires': requires
        }

    ## Write PEPs to CSV
    if write_peps_to_csv:
        keys = ['url', 'type', 'status', 'title', 'authors', 'created', 'replaces', 'superseded_by', 'requires']
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'peps.csv'), 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['number']+keys)
            for number,row in peps.items():
                csvwriter.writerow([number]+[row[key] for key in keys])

    ## Write authors to CSV
    if write_authors_to_csv:
        keys = ['first_name', 'last_name', 'suffix', 'email']
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'authors.csv'), 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['key']+keys)
            for author_key,row in authors_dict.items():
                csvwriter.writerow([author_key]+[row[key] for key in keys])

    return peps
