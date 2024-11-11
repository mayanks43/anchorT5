from sentence_transformers import util
import numpy as np
from enum import Enum
import torch
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

class RuleType(Enum):
    SYNTAX = 1
    SURFACE = 2

    def __str__(self):
        return '%s' % self.name.lower()

RULE_TYPE = RuleType.SURFACE # change this to switch between syntax and surface rules

USE_SYNTAX = True if RULE_TYPE == RuleType.SYNTAX else False
SUFFIX = ""

MODEL_SAVE_PATH = f"../nyt29_models/trained_on_anchor_word_{str(RULE_TYPE)}_rules{SUFFIX}"

# MODEL_SAVE_PATH = f"../nyt29_models/model_for_anchor_word_{str(RULE_TYPE)}_rules_bkgnd_pphrased{SUFFIX}"
# MODEL_SAVE_PATH = f"../final_data/anchor_word_trained_upto_{str(RULE_TYPE)}_rules{SUFFIX}"
# MODEL_SAVE_PATH = f"../combined/combined_model_for_anchor_word_{str(RULE_TYPE)}_rules{SUFFIX}"

MODEL_NAME = "Salesforce/codet5p-220m"
INPUT_MAX_LEN = 512 #input length
OUTPUT_MAX_LEN = 512 # output length

label_map = {
    "org:alternate_names": "an organization's alternate names",
    "org:city_of_headquarters": "an organization's city of headquarters",
    "org:country_of_headquarters": "an organization's country of headquarters",
    "org:dissolved": "an organization's date of dissolution",
    "org:founded": "an organization's date of founding",
    "org:founded_by": " an organization's founder",
    "org:member_of": "an organization's membership of another entity",
    "org:members": "an organization's members",
    "org:number_of_employees/members": "an organization's number of employees or members",
    "org:parents": "an organization's parents",
    "org:political/religious_affiliation": "an organization's political or religious affiliation",
    "org:shareholders": "an organization's shareholders",
    "org:stateorprovince_of_headquarters": "an organization's state or province of headquarters",
    "org:subsidiaries": "an organization's subsidiaries",
    "org:top_members/employees": "an organization's top members or employees",
    "org:website": "an organization's website",
    "per:age": "a person's age",
    "per:alternate_names": "a person's alternate names",
    "per:cause_of_death": "a person's cause of death",
    "per:charges": "a person's criminal charges",
    "per:children": "a person's children",
    "per:cities_of_residence": "a person's cities of residence",
    "per:city_of_birth": "a person's city of birth",
    "per:city_of_death": "a person's city of death",
    "per:countries_of_residence": "a person's countries of residence",
    "per:country_of_birth": "a person's country of birth",
    "per:country_of_death": "a person's country of death",
    "per:date_of_birth": "a person's date of birth",
    "per:date_of_death": "a person's date of death",
    "per:employee_of": "a person's employer",
    "per:origin": "a person's city or country of origin",
    "per:other_family": "a person's other family",
    "per:parents": "a person's parents",
    "per:religion": "a person's religion",
    "per:schools_attended": "schools attended by a person",
    "per:siblings": "a person's siblings",
    "per:spouse": "a person's spouse",
    "per:stateorprovince_of_birth": "a person's state or province of birth",
    "per:stateorprovince_of_death": "a person's state or province of death",
    "per:stateorprovinces_of_residence": "a person's state or province of residence",
    "per:title": "a person's title",
}

aux_verbs = {
    "is", "isn't", "be", "been", "being",
    "was", "wasn't",
    "has", "hasn't",
    "have", "haven't",
    "had", "hadn't",
    "shall", "shan't",
    "will", "won't",
    "should", "shouldn't",
    "could", "couldn't",
    "can", "can't",
    "do", "don't",
    "does", "doesn't",
    "did", "didn't",
    "are", "aren't",
    "were", "weren't",
    "may", "mayn't",
    "might", "mightn't",
    "must", "mustn't",
    "would", "wouldn't",
}

label_map_2_fstacred = {
    "org:alternate_names": "Alternate names of the organization including former names, aliases, alternate spellings, acronyms, abbreviations, translations or transliterations of names, and any official designators such as stock ticker code or airline call sign.",
    "org:city_of_headquarters": "Location of the headquarters of the organization at the city, town, or village level.",
    "org:country_of_headquarters": "Countries in which the headquarters of the organization are located.",
    "org:dissolved": "The date on which the organization was dissolved.",
    "org:founded": "The date on which the organization was founded.",
    "org:founded_by": "The person, organization, or geopolitical entity that founded the organization.",
    "org:member_of": "Organizations or geopolitical entities of which the organization is a member itself.",
    "org:members": "Organizations or Geopolitical entities that are members of the organization.",
    "org:number_of_employees/members": "The total number of people who are employed by or have membership in an organization.",
    "org:parents": "Organizations or geopolitical entities of which the organization is a subsidiary.",
    "org:political/religious_affiliation": "Ideological groups with which the organization is associated.",
    "org:shareholders": "Any organization, person, or geopolitical entity that holds shares (majority or not) of the organization.",
    "org:stateorprovince_of_headquarters": "Location of the headquarters of the organization at the state or province level.",
    "org:subsidiaries": "Organizations that are subsidiaries of the organization.",
    "org:top_members/employees": "The persons in high-level, leading positions at the organization.",
    "org:website": "An official top level URL for the organization's website.",
    "per:age": "A reported age of the person.",
    "per:alternate_names": "Names used to refer to the person that are distinct from the official name. Alternate names may include aliases, stage names, alternate transliterations, abbreviations, alternate spellings, nicknames, or birth names.",
    "per:cause_of_death": "The explicit cause of death for the person.",
    "per:charges": " The charges or crimes (alleged or convicted) of the person.",
    "per:children": "The children of the person, including adopted and step-children.",
    "per:cities_of_residence": "Geopolitical entities at the level of city, town, or village in which the person has lived.",
    "per:city_of_birth": " The geopolitical entity at the municipality level (city, town, or village) in which the person was born.",
    "per:city_of_death": "The geopolitical entity at the level of city, town, village in which the person died.",
    "per:countries_of_residence": "All countries in which the person has lived.",
    "per:country_of_birth": "The country in which the person was born.",
    "per:country_of_death": "The country in which the person died.",
    "per:date_of_birth": "The date on which the person was born.",
    "per:date_of_death": "The date of the person's death.",
    "per:employee_of": "The organizations or geopolitical entities (governments) by which the person has been employed.",
    "per:origin": "The nationality and/or ethnicity of the person.",
    "per:other_family": "Other family members of the person including brothers-in-law, sisters-in-law, grandparents, grandchildren, cousins, aunts, uncles, etc.",
    "per:parents": "The parents of the person.",
    "per:religion": "The religion to which the person has belonged.",
    "per:schools_attended": "Any school (college, high school, university, etc.) that the person has attended.",
    "per:siblings": "The brothers and sisters of the person.",
    "per:spouse": "The spouse or spouses of the person.",
    "per:stateorprovince_of_birth": "The geopolitical entity at state or province level in which the person was born.",
    "per:stateorprovince_of_death": "The geopolitical entity at state or province level in which the person died.",
    "per:stateorprovinces_of_residence": "Geopolitical entities at the state or province level in which the person has lived.",
    "per:title": " Official or unofficial name(s) of the employment or membership positions that have been held by the person.",
}

label_map_2_nyt29 = {
    "/location/administrative_division/country": "The country of an administrative division.",
    "/location/country/capital": "The capital city of a country.",
    "/location/country/administrative_divisions": "The administrative divisions within a country.",
    "/location/neighborhood/neighborhood_of": "The larger location a neighborhood is part of.",
    "/location/location/contains": "One location contains another.",
    "/people/person/nationality": "The nationality of a person.",
    "/people/person/place_lived": "Places where a person has lived.",
    "/people/deceased_person/place_of_death": "The place where a person died.",
    "/business/person/company": "The company a person is linked to.",
    "/location/us_state/capital": "The capital city of a US state.",
    "/people/person/place_of_birth": "The birthplace of a person.",
    "/people/person/children": "The children of a person.",
    "/business/company/founders": "The founders of a company.",
    "/business/company/place_founded": "The founding location of a company.",
    "/sports/sports_team/location": "The location or home of a sports team.",
    "/people/person/ethnicity": "The ethnicity of a person.",
    "/people/ethnicity/geographic_distribution": "Where people of a specific ethnicity are located.",
    "/people/person/religion": "The religion of a person.",
    "/business/company/major_shareholders": "The major shareholders of a company.",
    "/location/province/capital": "The capital of a province.",
    "/location/br_state/capital": "The capital of a Brazilian state.",
    "/business/company/advisors": "The advisors of a company.",
    "/film/film_location/featured_in_films": "Locations featured in films.",
    "/film/film/featured_film_locations": "Films that have showcased specific locations.",
    "/location/us_county/county_seat": "The administrative center of a US county.",
    "/time/event/locations": "The locations of events.",
    "/people/deceased_person/place_of_burial": "The burial place of a person.",
    "/people/place_of_interment/interred_here": "People interred at a specific location.",
    "/business/company_advisor/companies_advised": "The companies advised by a company advisor."
}

label_map_2 = {
    "org:alternate_names": "Alternate names of the organization including former names, aliases, alternate spellings, acronyms, abbreviations, translations or transliterations of names, and any official designators such as stock ticker code or airline call sign.",
    "org:city_of_headquarters": "Location of the headquarters of the organization at the city, town, or village level.",
    "org:country_of_headquarters": "Countries in which the headquarters of the organization are located.",
    "org:dissolved": "The date on which the organization was dissolved.",
    "org:founded": "The date on which the organization was founded.",
    "org:founded_by": "The person, organization, or geopolitical entity that founded the organization.",
    "org:member_of": "Organizations or geopolitical entities of which the organization is a member itself.",
    "org:members": "Organizations or Geopolitical entities that are members of the organization.",
    "org:number_of_employees/members": "The total number of people who are employed by or have membership in an organization.",
    "org:parents": "Organizations or geopolitical entities of which the organization is a subsidiary.",
    "org:political/religious_affiliation": "Ideological groups with which the organization is associated.",
    "org:shareholders": "Any organization, person, or geopolitical entity that holds shares (majority or not) of the organization.",
    "org:stateorprovince_of_headquarters": "Location of the headquarters of the organization at the state or province level.",
    "org:subsidiaries": "Organizations that are subsidiaries of the organization.",
    "org:top_members/employees": "The persons in high-level, leading positions at the organization.",
    "org:website": "An official top level URL for the organization's website.",
    "per:age": "A reported age of the person.",
    "per:alternate_names": "Names used to refer to the person that are distinct from the official name. Alternate names may include aliases, stage names, alternate transliterations, abbreviations, alternate spellings, nicknames, or birth names.",
    "per:cause_of_death": "The explicit cause of death for the person.",
    "per:charges": " The charges or crimes (alleged or convicted) of the person.",
    "per:children": "The children of the person, including adopted and step-children.",
    "per:cities_of_residence": "Geopolitical entities at the level of city, town, or village in which the person has lived.",
    "per:city_of_birth": " The geopolitical entity at the municipality level (city, town, or village) in which the person was born.",
    "per:city_of_death": "The geopolitical entity at the level of city, town, village in which the person died.",
    "per:countries_of_residence": "All countries in which the person has lived.",
    "per:country_of_birth": "The country in which the person was born.",
    "per:country_of_death": "The country in which the person died.",
    "per:date_of_birth": "The date on which the person was born.",
    "per:date_of_death": "The date of the person's death.",
    "per:employee_of": "The organizations or geopolitical entities (governments) by which the person has been employed.",
    "per:origin": "The nationality and/or ethnicity of the person.",
    "per:other_family": "Other family members of the person including brothers-in-law, sisters-in-law, grandparents, grandchildren, cousins, aunts, uncles, etc.",
    "per:parents": "The parents of the person.",
    "per:religion": "The religion to which the person has belonged.",
    "per:schools_attended": "Any school (college, high school, university, etc.) that the person has attended.",
    "per:siblings": "The brothers and sisters of the person.",
    "per:spouse": "The spouse or spouses of the person.",
    "per:stateorprovince_of_birth": "The geopolitical entity at state or province level in which the person was born.",
    "per:stateorprovince_of_death": "The geopolitical entity at state or province level in which the person died.",
    "per:stateorprovinces_of_residence": "Geopolitical entities at the state or province level in which the person has lived.",
    "per:title": " Official or unofficial name(s) of the employment or membership positions that have been held by the person.",
    "/location/administrative_division/country": "The country of an administrative division.",
    "/location/country/capital": "The capital city of a country.",
    "/location/country/administrative_divisions": "The administrative divisions within a country.",
    "/location/neighborhood/neighborhood_of": "The larger location a neighborhood is part of.",
    "/location/location/contains": "One location contains another.",
    "/people/person/nationality": "The nationality of a person.",
    "/people/person/place_lived": "Places where a person has lived.",
    "/people/deceased_person/place_of_death": "The place where a person died.",
    "/business/person/company": "The company a person is linked to.",
    "/location/us_state/capital": "The capital city of a US state.",
    "/people/person/place_of_birth": "The birthplace of a person.",
    "/people/person/children": "The children of a person.",
    "/business/company/founders": "The founders of a company.",
    "/business/company/place_founded": "The founding location of a company.",
    "/sports/sports_team/location": "The location or home of a sports team.",
    "/people/person/ethnicity": "The ethnicity of a person.",
    "/people/ethnicity/geographic_distribution": "Where people of a specific ethnicity are located.",
    "/people/person/religion": "The religion of a person.",
    "/business/company/major_shareholders": "The major shareholders of a company.",
    "/location/province/capital": "The capital of a province.",
    "/location/br_state/capital": "The capital of a Brazilian state.",
    "/business/company/advisors": "The advisors of a company.",
    "/film/film_location/featured_in_films": "Locations featured in films.",
    "/film/film/featured_film_locations": "Films that have showcased specific locations.",
    "/location/us_county/county_seat": "The administrative center of a US county.",
    "/time/event/locations": "The locations of events.",
    "/people/deceased_person/place_of_burial": "The burial place of a person.",
    "/people/place_of_interment/interred_here": "People interred at a specific location.",
    "/business/company_advisor/companies_advised": "The companies advised by a company advisor."
}

def surround_with_special_tokens(
    tokens,
    subjStart,
    subjEnd,
    objStart,
    objEnd,
    subjType,
    objType
):
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i == subjStart:
            new_tokens.extend(["<subj>", subjType.lower(), "</subj>"])
            i += subjEnd - subjStart + 1
        elif i == objStart:
            new_tokens.extend(["<obj>", objType.lower(),  "</obj>"])
            i += objEnd - objStart + 1
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

def add_types_to_sentence(
    tokens,
    subjStart,
    subjEnd,
    objStart,
    objEnd,
    subjType,
    objType
):
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i == subjStart:
            new_tokens.append(subjType)
            i += subjEnd - subjStart + 1
        elif i == objStart:
            new_tokens.append(objType)
            i += objEnd - objStart + 1
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

def get_synset_embeddings(model, word):
    synsets = wn.synsets(word)
    synset_names = [synset.name().split('.')[0] for synset in synsets]  # Extract synset names
    if not synset_names:  # If no synsets, default to the original word
        synset_names = [word]
    # Encode synset names and ensure output is 2D tensor for compatibility
    synset_embeddings = model.encode(synset_names, convert_to_tensor=True)
    return synset_embeddings

def process_tokens(model, phrase_embedding, tokens, similarity_threshold):
    max_similarities = []
    for token in tokens:
        synset_embeddings = get_synset_embeddings(model, token)
        # Ensure synset_embeddings is 2D, this should already be the case based on how embeddings are generated
        synset_embeddings_2d = synset_embeddings.unsqueeze(0) if synset_embeddings.dim() == 1 else synset_embeddings

        # Calculate cosine similarities
        sims = util.cos_sim(phrase_embedding, synset_embeddings_2d)[0]
        # Extract the maximum similarity value
        max_similarity = torch.max(sims).item()  # Convert to Python scalar for compatibility
        max_similarities.append(max_similarity)
    return max_similarities

def find_top_similar_words_and_generate_rules(
    stanza_model, sentence_transformer_model,
    sim_tokenizer, sim_model,
    phrase, tokens, pos_tags,
    subj_start, subj_end,
    obj_start, obj_end,
    subj_type, obj_type,
    top_k = 5, similarity_threshold = 0.25,
    orig_tokens = [],
    skip_pos_tag_check = False,
    skip_entities = False,
    skip_aux_verbs = False,
):
    final_rules = []
    if len(tokens) <= 2:
        if len(tokens) == 2:
            if subj_start < obj_start:
                rule = f"[word={subj_type}] [word={obj_type}]"
            else:
                rule = f"[word={obj_type}] [word={subj_type}]"
            final_rules.append(rule)
        else:
            raise Exception("Less than two tokens in sentence under consideration.")
        return final_rules

    # Encode just the phrase first
    phrase_embedding = sentence_transformer_model.encode([label_map_2[phrase]], convert_to_tensor=True)

    # print(phrase_embedding.size())
    # Process tokens to get their maximum cosine similarity
    max_similarities = process_tokens(sentence_transformer_model, phrase_embedding, tokens, similarity_threshold)

    # Convert max_similarities to a tensor for further processing
    max_similarities_tensor = torch.tensor(max_similarities)

    # Filter out words based on POS tags first if not skipping POS tag check
    if not skip_pos_tag_check:
        indices = [i for i, pos_tag in enumerate(pos_tags) if pos_tag.startswith(('N', 'V', 'R', 'J'))]
    else:
        indices = list(range(len(tokens)))

    # Filter tokens based on maximum similarity and POS tags, sort by similarity
    sorted_indices = np.argsort(max_similarities_tensor.cpu().numpy())[::-1]
    filtered_indices = [i for i in sorted_indices if i in indices and max_similarities_tensor[i] >= similarity_threshold][:top_k]

    # Process the final words to generate rules
    for index in filtered_indices:
        # skip first and last indices for syntax rules case
        if skip_pos_tag_check and (index == 0 or index == len(tokens) - 1):
            continue

        if skip_aux_verbs and tokens[index] in aux_verbs:
            continue

        if skip_entities:
            found_as_named_entity = False
            doc = stanza_model(" ".join(orig_tokens))
            # print("Entity check:", " ".join(orig_tokens))
            # for ent in doc.ents:
            #    print(ent.text, ent.type)
            for ent in doc.ents:
                if tokens[index] in ent.text.lower():
                    found_as_named_entity = True
                    break
            if found_as_named_entity:
                continue

        word = tokens[index]

        # lemma = stanza_model(word)[0].lemma_  # Process the word for lemma
        lemma = stanza_model(word).sentences[0].words[0].lemma.lower()

        # Determine the position of the word relative to the subject and object
        word_position = index  # Assuming index corresponds to the position in the original sentence

        if subj_end < word_position < obj_start or obj_end < word_position < subj_start:
            # Word is between subject and object
            if subj_end < word_position < obj_start:
                rule = f"[word={subj_type}] []* [lemma={lemma}] []* [word={obj_type}]"
            else:  # obj_end < word_position < subj_start
                rule = f"[word={obj_type}] []* [lemma={lemma}] []* [word={subj_type}]"
        elif word_position < subj_start and word_position < obj_start:
            # Word is before both subject and object
            if subj_start < obj_start:
                # Subject comes before object
                rule = f"[lemma={lemma}] []* [word={subj_type}] []* [word={obj_type}]"
            else:
                # Object comes before subject
                rule = f"[lemma={lemma}] []* [word={obj_type}] []* [word={subj_type}]"
        elif word_position > subj_end and word_position > obj_end:
            # Word is after both subject and object
            if subj_end < obj_end:
                # Subject comes before object
                rule = f"[word={subj_type}] []* [word={obj_type}] []* [lemma={lemma}]"
            else:
                # Object comes before subject
                rule = f"[word={obj_type}] []* [word={subj_type}] []* [lemma={lemma}]"
        else:
            # Handle edge cases or ambiguities if needed
            continue

        # final_rules.append((index, lemma, rule, similarity))
        final_rules.append(rule)

        # final_rules.extend(rules)

    return final_rules

def gen_input_text(
    data,
    idx,
    get_rules = False,
    use_syntax_path_tokens = False,
    train = False
):
    support = data[idx][0]
    relation_description = label_map_2[support["relation"]]
    sentence = support["sentence"]

    text = "Generate relation extraction rule for relation: "
    text += relation_description + " Given sentence: " + (
        " ".join(surround_with_special_tokens(
            [x.lower() for x in sentence["tokens"]],
            subjStart = sentence["subjStart"],
            subjEnd= sentence["subjEnd"],
            objStart = sentence["objStart"],
            objEnd = sentence["objEnd"],
            subjType = sentence["subjType"],
            objType = sentence["objType"]
        ))
    )

    if use_syntax_path_tokens:
        if train:
            tokens_on_syntax_path = data[idx][3]
        else:
            tokens_on_syntax_path = data[idx][2]
        text += " Given tokens on syntax path: " + " ".join(tokens_on_syntax_path) + "."

    text += " The rules are: "

    if get_rules:
        rule_dicts = data[idx][1]
        if rule_dicts:
            rules = " ~ ".join([x["pattern"] for x in rule_dicts])
        else:
            rules = "No possible rule."
        return text, rules, support
    else:
        return text, support

def gen_input_text2(
    data,
    get_rules = False,
    use_syntax_path_tokens = False,
    train = False
):
    support = data[0]
    relation_description = label_map_2[support["relation"]]
    sentence = support["sentence"]

    text = "Generate relation extraction rule for relation: "
    text += relation_description + " Given sentence: " + (
        " ".join(surround_with_special_tokens(
            [x.lower() for x in sentence["tokens"]],
            subjStart = sentence["subjStart"],
            subjEnd= sentence["subjEnd"],
            objStart = sentence["objStart"],
            objEnd = sentence["objEnd"],
            subjType = sentence["subjType"],
            objType = sentence["objType"]
        ))
    )

    if use_syntax_path_tokens:
        if train:
            tokens_on_syntax_path = data[3]
        else:
            tokens_on_syntax_path = data[2]
        text += " Given tokens on syntax path: " + " ".join(tokens_on_syntax_path) + "."

    text += " The rules are: "

    if get_rules:
        rule_dicts = data[1]
        if rule_dicts:
            rules = " ~ ".join([x["pattern"] for x in rule_dicts])
        else:
            rules = "No possible rule."
        return text, rules, support
    else:
        return text, support

def gen_pphrase_input_text(
    lower_tokens,
    relation_description,
    subj_start,
    subj_end,
    obj_start,
    obj_end,
    subj_type,
    obj_type,
    use_syntax_path_tokens = False,
    tokens_on_syntax_path = [],
):
    text = "Generate relation extraction rule for relation: "
    text += relation_description + " Given sentence: " + (
        " ".join(surround_with_special_tokens(
            lower_tokens,
            subj_start,
            subj_end,
            obj_start,
            obj_end,
            subj_type,
            obj_type
        ))
    )

    if use_syntax_path_tokens:
        text += " Given tokens on syntax path: " + " ".join(tokens_on_syntax_path) + "."

    text += " The rules are: "

    return text
