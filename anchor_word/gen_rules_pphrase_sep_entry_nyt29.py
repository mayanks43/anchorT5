from transformers import AutoTokenizer, RobertaTokenizer, RobertaForMaskedLM
from sentence_transformers import SentenceTransformer
import pytorch_lightning as pl
import warnings
import json
from tqdm import tqdm
from utils import (
    find_top_similar_words_and_generate_rules,
    label_map_2,
    add_types_to_sentence,
)
import argparse
import stanza
import copy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split_type', help="Data split type to use")
    parser.add_argument('-k', '--top_k', type=int, help="Number of rules")
    args = parser.parse_args()

    SPLIT = args.split_type
    TOP_K = args.top_k

    SYNTAX_OUTPUT_FILE = SPLIT + "_upto_" + str(TOP_K) + "_anchor_syntax_rules_pphrase_sep_entry.jsonl"
    SURFACE_OUTPUT_FILE = SPLIT + "_upto_" + str(TOP_K) + "_anchor_surface_rules_pphrase_sep_entry.jsonl"

    # DIRECTORY = "../final_data/"
    # PATH = "/workspace/fstacred/data_few_shot/_" + SPLIT + "_data_syntax_path.jsonl"
    # pphrases = json.load(open(f"/workspace/pphrase/{SPLIT}_pphrase_with_ents_syntax.json", "r"))

    DIRECTORY = "../nyt29/"
    PATH = "/workspace/nyt29/_" + SPLIT + "_data_syntax_path.jsonl"
    pphrases = json.load(open(f"/workspace/nyt29_pphrase/{SPLIT}_pphrase_with_ents_syntax.json", "r"))

    SYNTAX_RULE_TYPE = "org.clulab.odinsynth.evaluation.genericeval.SimplifiedSyntaxRuleType"
    SURFACE_RULE_TYPE = "org.clulab.odinsynth.evaluation.genericeval.SurfaceRuleType"

    def prepare_patterns(rules, rel_name, rule_type, direc):
        patterns = []
        for pred in rules:
            rule = pred
            pattern = {}
            pattern["pattern"] = rule
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
    for idx in tqdm(range(len(sentences))):
        support_sentence = sentences[idx][0]
        sentence_with_types = sentences[idx][1]
        tokens_on_syntax_path = sentences[idx][2]

        relation_name = support_sentence["relation"]
        sentence = support_sentence["sentence"]
        doc_id = sentence["id"]

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
        syntax_patterns = prepare_patterns(
            syntax_rules,
            relation_name,
            SYNTAX_RULE_TYPE,
            direction,
        )
        syntax_results.append([
            support_sentence,
            syntax_patterns,
            sentence_with_types,
            tokens_on_syntax_path,
        ])

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
        surface_patterns = prepare_patterns(
            surface_rules,
            relation_name,
            SURFACE_RULE_TYPE,
            direction,
        )
        surface_results.append([
            support_sentence,
            surface_patterns,
            sentence_with_types,
        ])

        pphrase_idx = 0
        for pphrase in pphrases.get(relation_name + "#" + doc_id.strip('"'), []):
            # print(doc_id.strip('"'))
            pphrase_subj_start = pphrase["subj_int"][0]
            pphrase_obj_start = pphrase["obj_int"][0]
            if "tokens_on_syntax_path" not in pphrase:
                print(
                    "Missing tokens on syntax path",
                    relation_name + "#" + doc_id.strip('"'),
                    pphrase
                )
                continue
            pphrase_tokens_on_syntax_path = pphrase["tokens_on_syntax_path"]
            pphrase_sentence_with_types = add_types_to_sentence(
                pphrase["tokens"],
                pphrase["subj_int"][0],
                pphrase["subj_int"][1],
                pphrase["obj_int"][0],
                pphrase["obj_int"][1],
                sentence["subjType"],
                sentence["objType"],
            )
            pphrase_support_sentence = copy.deepcopy(support_sentence)
            pphrase_support_sentence["sentence"]["tokens"] = pphrase["tokens"]
            pphrase_support_sentence["sentence"]["id"] = "\"" + sentence["id"].strip('"') + "#" + str(pphrase_idx) + "\""
            pphrase_support_sentence["sentence"]["docId"] = "\"" + sentence["docId"].strip('"') + "#" + str(pphrase_idx) + "\""
            pphrase_support_sentence["sentence"]["subjStart"] = pphrase["subj_int"][0]
            pphrase_support_sentence["sentence"]["subjEnd"] = pphrase["subj_int"][1]
            pphrase_support_sentence["sentence"]["objStart"] = pphrase["obj_int"][0]
            pphrase_support_sentence["sentence"]["objEnd"] = pphrase["obj_int"][1]
            pphrase_support_sentence["sentence"]["posTags"] = pphrase["pos"]

            assert pphrase_support_sentence["sentence"]["subjType"] == pphrase["subj_type"]
            assert pphrase_support_sentence["sentence"]["objType"] == pphrase["obj_type"]

            if (pphrase_subj_start < pphrase_obj_start):
                pphrase_direction = "org.clulab.odinsynth.evaluation.genericeval.SubjObjDirection"
                pphrase_syn_subj_start = 0
                pphrase_syn_subj_end = 0
                pphrase_syn_obj_start = len(pphrase_tokens_on_syntax_path) - 1
                pphrase_syn_obj_end = len(pphrase_tokens_on_syntax_path) - 1
            else:
                pphrase_direction = "org.clulab.odinsynth.evaluation.genericeval.ObjSubjDirection"
                pphrase_syn_subj_start = len(pphrase_tokens_on_syntax_path) - 1
                pphrase_syn_subj_end = len(pphrase_tokens_on_syntax_path) - 1
                pphrase_syn_obj_start = 0
                pphrase_syn_obj_end = 0

            pphrase_syntax_rules = find_top_similar_words_and_generate_rules(
                stanza_model,
                sentence_transformer_model,
                roberta_tokenizer,
                roberta_model,
                relation_name,
                [token.lower() for token in pphrase_tokens_on_syntax_path],
                pphrase["pos"],
                pphrase_syn_subj_start,
                pphrase_syn_subj_end,
                pphrase_syn_obj_start,
                pphrase_syn_obj_end,
                sentence["subjType"].lower(),
                sentence["objType"].lower(),
                skip_pos_tag_check = True,
                top_k = TOP_K,
                orig_tokens = pphrase["tokens"],
                skip_entities = False,
                skip_aux_verbs = True,
            )
            pphrase_syntax_patterns = prepare_patterns(
                pphrase_syntax_rules,
                relation_name,
                SYNTAX_RULE_TYPE,
                pphrase_direction,
            )
            assert [token.lower() for token in pphrase_tokens_on_syntax_path] == pphrase_tokens_on_syntax_path
            syntax_results.append([
                pphrase_support_sentence,
                pphrase_syntax_patterns,
                pphrase_sentence_with_types,
                pphrase_tokens_on_syntax_path
            ])

            pphrase_surface_rules = find_top_similar_words_and_generate_rules(
                stanza_model,
                sentence_transformer_model,
                roberta_tokenizer,
                roberta_model,
                relation_name,
                [token.lower() for token in pphrase["tokens"]],
                pphrase["pos"],
                pphrase["subj_int"][0],
                pphrase["subj_int"][1],
                pphrase["obj_int"][0],
                pphrase["obj_int"][1],
                sentence["subjType"].lower(),
                sentence["objType"].lower(),
                top_k = TOP_K,
                orig_tokens = pphrase["tokens"],
                skip_entities = False,
                skip_aux_verbs = True,
            )
            pphrase_surface_patterns = prepare_patterns(
                pphrase_surface_rules,
                relation_name,
                SURFACE_RULE_TYPE,
                pphrase_direction,
            )
            surface_results.append([
                pphrase_support_sentence,
                pphrase_surface_patterns,
                pphrase_sentence_with_types,
            ])
            pphrase_idx += 1

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
