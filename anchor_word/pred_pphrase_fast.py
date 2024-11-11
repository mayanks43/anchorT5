import json
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import pytorch_lightning as pl
import warnings

# Imports from local modules
from utils import (
    gen_pphrase_input_text,
    RuleType,
    RULE_TYPE,
    USE_SYNTAX,
    MODEL_NAME,
    INPUT_MAX_LEN,
    OUTPUT_MAX_LEN,
    MODEL_SAVE_PATH,
    label_map_2,
    gen_input_text2
)
from model import TextRuleT5Model

# Constants

# OUTPUT_FILE = f"../final_data/trained_on_upto_{NUM_SEQS}_{RULE_TYPE}_results_with_pphrases_anchor_words.json"
# TEST_PATH = "/home/cc/fstacred/data_few_shot/_test_data_syntax_path.jsonl"
# TEST_PPHRASES = "/home/cc/pphrase/test_pphrase_with_ents_syntax.json"

OUTPUT_FILE = f"../nyt29/model_gen_anchor_{RULE_TYPE}_rules_with_support_pphrased.jsonl"
TEST_PATH = "/workspace/nyt29/_test_data_syntax_path.jsonl"
TEST_PPHRASES_PATH = "/workspace/nyt29_pphrase/test_pphrase_with_ents_syntax.json"

NUM_BEAMS = 4
BATCH_SIZE = 1

# Determine rule type based on configuration
syntax_rule_type = "org.clulab.odinsynth.evaluation.genericeval.SimplifiedSyntaxRuleType"
surface_rule_type = "org.clulab.odinsynth.evaluation.genericeval.SurfaceRuleType"
SCALA_RULE_TYPE_VALUE = syntax_rule_type if RULE_TYPE == RuleType.SYNTAX else surface_rule_type

def generate_rule(batch_texts, trained_model, tokenizer):
    inputs_encoding = tokenizer(
        batch_texts,
        max_length=INPUT_MAX_LEN,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    device = next(trained_model.parameters()).device
    inputs_encoding["input_ids"] = inputs_encoding["input_ids"].to(device)
    inputs_encoding["attention_mask"] = inputs_encoding["attention_mask"].to(device)

    generate_ids = trained_model.model.generate(
        input_ids=inputs_encoding["input_ids"],
        attention_mask=inputs_encoding["attention_mask"],
        max_length=OUTPUT_MAX_LEN,
        num_beams=NUM_BEAMS,
        num_return_sequences=1,
        early_stopping=False,
    )

    preds = [
        tokenizer.decode(
            gen_id,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        for gen_id in generate_ids
    ]

    return preds  # Returns a list of predictions for the batch

torch.set_float32_matmul_precision('medium')
pl.seed_everything(100)
warnings.filterwarnings("ignore")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load data
print("Loading data")
data = [json.loads(line) for line in open(TEST_PATH, "r")]
test_pphrases = json.load(open(TEST_PPHRASES_PATH, "r"))

# Load model
print("Loading model", MODEL_SAVE_PATH + '.ckpt')
trained_model = TextRuleT5Model.load_from_checkpoint(MODEL_SAVE_PATH + '.ckpt')
trained_model.freeze()

# Process data in batches
results = []
print("Generating predictions")
for i in tqdm(range(0, len(data), BATCH_SIZE)):
    batch_data = data[i:i + BATCH_SIZE]
    batch_texts = []
    paraphrase_texts = []

    for data_point in batch_data:
        text, support = gen_input_text2(
            data_point,
            get_rules=False,
            use_syntax_path_tokens=USE_SYNTAX,
            train=False
        )
        batch_texts.append(text)
        doc_id = support["sentence"]["id"]
        for pphrase in test_pphrases.get(doc_id.strip('"'), []):
            if "tokens_on_syntax_path" in pphrase:
                paraphrase_text = gen_pphrase_input_text(
                    [token.lower() for token in pphrase["tokens"]],
                    label_map_2[support["relation"]],
                    pphrase["subj_int"][0],
                    pphrase["subj_int"][1],
                    pphrase["obj_int"][0],
                    pphrase["obj_int"][1],
                    support["sentence"]["subjType"],
                    support["sentence"]["objType"],
                    True,
                    pphrase["tokens_on_syntax_path"]
                )
                #print([token.lower() for token in pphrase["tokens"]])
                #print(paraphrase_text)
                paraphrase_texts.append(paraphrase_text)

    all_texts = batch_texts + paraphrase_texts
    outputs = generate_rule(all_texts, trained_model, tokenizer)

    primary_outputs = outputs[:len(batch_texts)]
    paraphrase_outputs = outputs[len(batch_texts):]

    paraphrase_index = 0
    for idx, output in enumerate(primary_outputs):
        preds = [] if "No possible rule." in output else output.split(" ~ ")
        support = batch_data[idx][0]
        pphrases = test_pphrases.get(support["sentence"]["id"].strip('"'), [])

        pphrase_preds = []
        for _ in pphrases:
            p_output = paraphrase_outputs[paraphrase_index]
            paraphrase_index += 1
            if "No possible rule." not in p_output:
                pphrase_preds.extend(p_output.split(" ~ "))
        preds.extend(pphrase_preds)
        direction = "org.clulab.odinsynth.evaluation.genericeval.SubjObjDirection" if support["sentence"]["subjStart"] < support["sentence"]["objStart"] else "org.clulab.odinsynth.evaluation.genericeval.ObjSubjDirection"

        patterns = [{
            "pattern": pred,
            "relation": support["relation"],
            "direction": {"$type": direction},
            "weight": 1,
            "patternType": {"$type": SCALA_RULE_TYPE_VALUE}
        } for pred in list(set(preds))]

        gen_rule = [support, patterns]
        results.append(gen_rule)

# Save results
with open(OUTPUT_FILE, "w") as file:
    for result in results:
        file.write(json.dumps(result) + "\n")

print("Wrote to:", OUTPUT_FILE)
