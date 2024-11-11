from transformers import AutoTokenizer, RobertaTokenizer, RobertaForMaskedLM
from sentence_transformers import SentenceTransformer
import pytorch_lightning as pl
import warnings
import json
from tqdm import tqdm
from utils import (
    find_top_similar_words_and_generate_rules,
    label_map_2,
)
import argparse
import stanza

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split_type', help="Data split type to use")
    parser.add_argument('-k', '--top_k', type=int, help="Number of rules")
    args = parser.parse_args()

    SPLIT = args.split_type
    TOP_K = args.top_k
    DIRECTORY = "../final_data/"
    SYNTAX_OUTPUT_FILE = SPLIT + "_upto_" + str(TOP_K) + "_anchor_syntax_rules.jsonl"
    SURFACE_OUTPUT_FILE = SPLIT + "_upto_" + str(TOP_K) + "_anchor_surface_rules.jsonl"

    PATH = "/workspace/fstacred/data_few_shot/_" + SPLIT + "_data_syntax_path.jsonl"
    SYNTAX_RULE_TYPE = "org.clulab.odinsynth.evaluation.genericeval.SimplifiedSyntaxRuleType"
    SURFACE_RULE_TYPE = "org.clulab.odinsynth.evaluation.genericeval.SurfaceRuleType"

    def prepare_patterns(rules, rel_name, direc, rule_type):
        patterns = []
        for pred in rules:
            pattern = {}
            pattern["pattern"] = pred
            pattern["relation"] = rel_name
            pattern["direction"] = {"$type": direc}
            pattern["weight"] = 1
            pattern["patternType"] = {"$type": rule_type}
            patterns.append(pattern)
        return patterns

    stanza_model = stanza.Pipeline(lang="en", tokenize_pretokenized=True)

    # Load a pre-trained sentence-transformer model
    sentence_transformer_model = SentenceTransformer("all-MiniLM-L12-v2")
    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    roberta_model = RobertaForMaskedLM.from_pretrained("roberta-base")

    print("Loading data")
    sentences = []
    with open(PATH, "r") as file:
        for line in file:
            json_obj = json.loads(line)
            sentences.append(json_obj)

    print("Getting rules")
    syntax_results = []
    surface_results = []
    for i in tqdm(range(len(sentences))):
        support_sentence = sentences[i][0]
        sentence_with_types = sentences[i][1]
        tokens_on_syntax_path = sentences[i][2]

        relation_name = support_sentence["relation"]
        sentence = support_sentence["sentence"]

        if (sentence["subjStart"] < sentence["objStart"]):
            direction = "org.clulab.odinsynth.evaluation.genericeval.SubjObjDirection"
            syn_subj_start = 0
            syn_subj_end = 0
            syn_obj_start = len(tokens_on_syntax_path) - 1
            syn_obj_end = len(tokens_on_syntax_path) - 1
        else:
            direction = "org.clulab.odinsynth.evaluation.genericeval.ObjSubjDirection"
            syn_subj_start = len(tokens_on_syntax_path) - 1
            syn_subj_end = len(tokens_on_syntax_path) - 1
            syn_obj_start = 0
            syn_obj_end = 0

        syntax_rules = find_top_similar_words_and_generate_rules(
            stanza_model,
            sentence_transformer_model,
            roberta_tokenizer,
            roberta_model,
            relation_name,
            tokens_on_syntax_path,
            sentence["posTags"],
            syn_subj_start,
            syn_subj_end,
            syn_obj_start,
            syn_obj_end,
            sentence["subjType"].lower(),
            sentence["objType"].lower(),
            skip_pos_tag_check = True,
            top_k = TOP_K,
            orig_tokens = sentence["tokens"],
            skip_entities = False,
            skip_aux_verbs = True,
        )

        surface_rules = find_top_similar_words_and_generate_rules(
            stanza_model,
            sentence_transformer_model,
            roberta_tokenizer,
            roberta_model,
            relation_name,
            [x.lower() for x in sentence["tokens"]],
            sentence["posTags"],
            sentence["subjStart"],
            sentence["subjEnd"],
            sentence["objStart"],
            sentence["objEnd"],
            sentence["subjType"].lower(),
            sentence["objType"].lower(),
            top_k = TOP_K,
            orig_tokens = sentence["tokens"],
            skip_entities = False,
            skip_aux_verbs = True,
        )

        syntax_patterns = prepare_patterns(
            syntax_rules,
            relation_name,
            direction,
            SYNTAX_RULE_TYPE,
        )
        surface_patterns = prepare_patterns(
            surface_rules, # + ngram_surface_rules,
            relation_name,
            direction,
            SURFACE_RULE_TYPE,
        )

        syntax_results.append([
            support_sentence,
            syntax_patterns,
            sentence_with_types,
            tokens_on_syntax_path,
        ])
        surface_results.append([
            support_sentence,
            surface_patterns,
            sentence_with_types,
        ])

        if i % 100 == 0:
            print("support_sentence: ", support_sentence)
            print("tokens_on_syntax_path: ", tokens_on_syntax_path)
            print("syntax_patterns: ", syntax_patterns)
            print("surface_patterns: ", surface_patterns)

    with open(DIRECTORY + SYNTAX_OUTPUT_FILE, "w") as file:
        for result in syntax_results:
            file.write(json.dumps(result) + "\n")
    print("Wrote to: ", SYNTAX_OUTPUT_FILE)

    with open(DIRECTORY + "deter_" + SYNTAX_OUTPUT_FILE, "w") as file:
        for result in syntax_results:
            file.write(json.dumps([result[0], result[1]]) + "\n")
    print("Wrote to: ", "deter_" + SYNTAX_OUTPUT_FILE)

    with open(DIRECTORY + SURFACE_OUTPUT_FILE, "w") as file:
        for result in surface_results:
            file.write(json.dumps(result) + "\n")
    print("Wrote to: ", SURFACE_OUTPUT_FILE)

    with open(DIRECTORY + "deter_" + SURFACE_OUTPUT_FILE, "w") as file:
        for result in surface_results:
            file.write(json.dumps([result[0], result[1]]) + "\n")
    print("Wrote to: ", "deter_" + SURFACE_OUTPUT_FILE)
