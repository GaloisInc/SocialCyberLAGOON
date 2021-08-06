import re
import unicodedata


########################################################################
# Single name transformations
########################################################################

def rearrange_name(name, separators=None):
    separators = separators or [' - ', ', ']
    for sep in separators:
        if sep in name:
            names = name.split(sep)
            if len(names) > 2:
                continue
            name = names[-1] + ' ' + names[0]
    return name

def remove_brackets(name):
    name = name.replace("(","").replace(")","")
    return name

def remove_bracketed_terms(name):
    name = re.sub(r"\(.*?\)", "", name)
    name = name.replace("  ", " ") #since removing a bracketed term from the middle of the string will leave 2 spaces
    return name

def remove_double_quotes(name):
    name = name.replace('"', '')
    return name

def remove_double_quoted_terms(name):
    name = re.sub(r'".*?"', '', name)
    name = name.replace("  ", " ") #since removing a quoted term from the middle of the string will leave 2 spaces
    return name

def remove_titles(name, titles=None):
    titles = titles or ['mrs. ', 'mr. ', 'ms. ', 'dr. ']
    if name.startswith("reverend "):
        name = name[9:]
    elif name.startswith("rev. "):
        name = name[5:]
    for title in titles:
        if name.startswith(title):
            name = name.replace(title, "", 1)
    return name

def process_junior(name):
    if name.endswith(", jr."):
        name = name.replace(", jr.", " jr")
    elif name.endswith("jr."):
        name = name[:-1]
    return name

def process_saint(name):
    if name.startswith("st. "):
        name = name.replace("st. ", "st ", 1)
    return name

def remove_company_suffix(name, suffixes=None):
    suffixes = suffixes or ['inc', 'co. ltd','co. limited','co ltd','co limited', 'pvt. ltd','pvt. limited','pvt ltd','pvt limited', 'private ltd','private limited', 'ltd', 'limited', 'gmbh']
    for suffix in suffixes:
        if name.endswith(f", {suffix}."):
            name = name[:-(len(suffix)+3)]
        elif name.endswith(f" {suffix}."):
            name = name[:-(len(suffix)+2)]
        elif name.endswith(f", {suffix}"):
            name = name[:-(len(suffix)+2)]
        elif name.endswith(f" {suffix}"):
            name = name[:-(len(suffix)+1)]
    return name

def remove_family(name, terms=None):
    terms = terms or [' family members.', ' family members', ' family.', ' family']
    for term in terms:
        if name.endswith(term):
            name = name[:-len(term)]
    return name

def process_initials(name):
    if len(name) <= 1:
        return name
    
    capitals = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    output = ""
    i = 0
    while i < len(name):
        if i==0:
            if name[i] in capitals and name[i+1] == ".":
                output += (name[i] + ' ')
                i += 2
            else:
                output += name[i]
                i += 1
        elif i != len(name)-1:
            if name[i-1] not in capitals and name[i] in capitals and name[i+1] == ".":
                output += (name[i] + ' ')
                i += 2
            else:
                output += name[i]
                i += 1
        else:
            output += name[i]
            i += 1
    output = output.replace("  ", " ")
    return output

def remove_middle_names(name):
    name = re.sub(" .* ", " ", name)
    return name

def process_special_characters(name, extra_replacements=None):
    extra_replacements = extra_replacements or {'ð':'d', 'ł':'l'}
    for rep in extra_replacements:
        name = name.replace(rep,extra_replacements[rep])
    name = unicodedata.normalize('NFKD',name).encode('ascii','ignore').decode()
    return name

def process_and(name):
    name = name.replace(" & ", " and ")
    return name

def remove_leading_trailing_spaces(name):
    name = name.lstrip().rstrip()
    return name


########################################################################
# Group transformations
########################################################################

def get_name_from_item(item, separators=None):
    """
    The complete item can read as 'Lionel Messi (star), Argentine footballer'
    We just want the part before the comma / semicolon / colon and without the brackets, i.e. Lionel Messi
    """
    separators = separators or [',', ';', ':']
    indexes = []
    for sep in separators:
        index = item.find(sep)
        indexes.append(index if index != -1 else len(item))
    
    name = item[:min(indexes)]
    return name

def split_and_complete_names(sentence):
    clauses = sentence.split(' and ')
    if len(clauses) != 2:
        return clauses
    if clauses[-1].startswith(('his','her')):
        return clauses
    last_clause_names = clauses[-1].split(' ')
    for i,name in enumerate(last_clause_names):
        if i==0:
            continue
        if i==len(clauses[0].split(' ')):
            clauses[0] += f' {name}'
    return clauses


########################################################################
# Tests
########################################################################
def tests():

    assert remove_brackets("(CEO) Lionel Messi (star)") == "CEO Lionel Messi star"
    assert remove_brackets("CEO Lionel Messi (star)") == "CEO Lionel Messi star"
    assert remove_brackets("(CEO Lionel Messi star)") == "CEO Lionel Messi star"
    
    assert remove_bracketed_terms("CEO (Lionel Messi) star") == "CEO star"
    assert remove_bracketed_terms("(CEO) Lionel Messi (star)") == " Lionel Messi "
    assert remove_bracketed_terms("(CEO Lionel Messi star)") == ""
    assert remove_bracketed_terms("(CEO) (Lionel) Messi star") == " Messi star"
    assert remove_bracketed_terms("CEO Lionel (Messi) (star)") == "CEO Lionel "
    assert remove_bracketed_terms("(CEO Lionel) (Messi star)") == " "

    assert remove_titles("mrs. somebody") == "somebody"
    assert remove_titles("dr. somebody") == "somebody"
    assert remove_titles("rev. dr. somebody") == "somebody"
    assert remove_titles("reverend somebody") == "somebody"

    assert process_junior("Aluizio Prata, jr.") == "Aluizio Prata jr"
    assert process_junior("Aluizio Prata jr.") == "Aluizio Prata jr"
    assert process_junior("Aluizio Prata jr") == "Aluizio Prata jr"
    
    assert process_saint("st. Clare") == "st Clare"
    assert process_saint("st Clare") == "st Clare"

    assert remove_company_suffix("Galois, inc.") == "Galois"
    assert remove_company_suffix("Galois inc.") == "Galois"
    assert remove_company_suffix("Galois co. limited") == "Galois"
    assert remove_company_suffix("Galois, private ltd.") == "Galois"
    assert remove_company_suffix("Galois pvt. ltd") == "Galois"

    assert remove_family("Sarasola Marulanda family.") == "Sarasola Marulanda"
    assert remove_family("Nakash family members") == "Nakash"

    assert process_initials("A.Singh") == "A Singh"
    assert process_initials("A. Singh") == "A Singh"
    assert process_initials("A Singh") == "A Singh"
    assert process_initials("A.B.C.Singh") == "A B C Singh"
    assert process_initials("A.B.C. Singh") == "A B C Singh"
    assert process_initials("ABC.Singh") == "ABC.Singh"
    assert process_initials("ABC. Singh") == "ABC. Singh"
    assert process_initials("A B C Singh") == "A B C Singh"
    assert process_initials("ABC Singh") == "ABC Singh"
    assert process_initials("Singh INC.") == "Singh INC." #multiple capital letters and dot are left untouched

    assert remove_double_quotes('Maria Imelda "Imee" Marcos Manotoc') == 'Maria Imelda Imee Marcos Manotoc'
    assert remove_double_quotes('''"Mar'ia ""''') == "Mar'ia "
    
    assert remove_double_quoted_terms('Maria Imelda "Imee" Marcos Manotoc') == 'Maria Imelda Marcos Manotoc'
    assert remove_double_quoted_terms('Maria "Imelda" Imee "Marcos" Manotoc') == 'Maria Imee Manotoc'
    assert remove_double_quoted_terms('Maria "Imelda Imee Marcos" Manotoc') == 'Maria Manotoc'

    assert remove_middle_names("Sean Mulryan") == "Sean Mulryan"
    assert remove_middle_names("Luca Cordero di Montezemolo") == "Luca Montezemolo"
    assert remove_middle_names("Hamad bin Jassim bin Jaber Al Thani") == "Hamad Thani"
    assert remove_middle_names("K. P. Singh") == "K. Singh"
    assert remove_middle_names("Charles Stuart de'Moivre") == "Charles de'Moivre"

    assert process_special_characters("sigmundur davíð gunnlaugsson") == "sigmundur david gunnlaugsson"
    assert process_special_characters("Çaðłayan") == "Cadlayan"

    assert process_and("X & Y") == "X and Y"
    assert process_and("X& Y") == "X& Y"

    assert remove_leading_trailing_spaces("  Lionel Messi ") == "Lionel Messi"

    assert get_name_from_item("Helene Mathieu; legal consultant based in Dubai, member of the Quebec Bar, worked with Mossack Fonseca to form shell companies") == "Helene Mathieu"
    assert get_name_from_item("Shishir Bajoria, Indian promoter of SK Bajoria Group; which has steel refractory units") == "Shishir Bajoria"
    assert get_name_from_item("Bank Leumi's Israeli bank: representatives and board members") == "Bank Leumi's Israeli bank"
    assert get_name_from_item("Luca Cordero di Montezemolo Italian businessman and politician") == "Luca Cordero di Montezemolo Italian businessman and politician"

    assert split_and_complete_names('Rajiv Dey and his sister Priya Dey') == ['Rajiv Dey', 'his sister Priya Dey']
    assert split_and_complete_names('Priyanka Dey and her sisters Priya Dey and others') == ['Priyanka Dey', 'her sisters Priya Dey', 'others']
    assert split_and_complete_names('Francisco and Juan Jose Franco Suelves') == ['Francisco Jose Franco Suelves', 'Juan Jose Franco Suelves']
    assert split_and_complete_names('Francisco Ramon and Juan Jose Franco Suelves') == ['Francisco Ramon Franco Suelves', 'Juan Jose Franco Suelves']
    
    assert rearrange_name("Terrero - Santiago") == "Santiago Terrero"
    assert rearrange_name("Terrill - Brendan Mark") == "Brendan Mark Terrill"
    assert rearrange_name("Terrill, Brendan Mark") == "Brendan Mark Terrill"
    assert rearrange_name("Andre Terrill, Brendan Mark") == "Brendan Mark Andre Terrill"
    assert rearrange_name("Terrill Brendan Mark") == "Terrill Brendan Mark"


########################################################################
# Examples of putting these together
########################################################################

def single_name_transformations(name):
    ''' Commented out ones are too aggressive and may lead to loss of useful information '''
    name = process_special_characters(name)
    name = process_initials(name)
    name = name.lower()

    name = remove_brackets(name)
    # name = remove_bracketed_terms(name)
    name = remove_double_quotes(name)
    # name = remove_double_quoted_terms(name)

    name = remove_titles(name)
    name = process_junior(name)
    name = process_saint(name)
    name = remove_company_suffix(name)
    name = remove_family(name)
    # name = remove_middle_names(name)

    name = process_and(name)
    name = remove_leading_trailing_spaces(name)
    
    return name

def process_items(items):
    names = []
    for item in items:
        sentence = get_name_from_item(item)
        clauses = split_and_complete_names(sentence)
        names.extend(clauses)
    
    names = [single_name_transformations(name) for name in names]
    return names
