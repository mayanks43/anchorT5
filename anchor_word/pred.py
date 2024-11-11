from transformers import AutoTokenizer
import pytorch_lightning as pl
import warnings
import json
from tqdm import tqdm
from utils import (
    gen_input_text,
    RuleType,
    RULE_TYPE,
    USE_SYNTAX,
    MODEL_NAME,
    INPUT_MAX_LEN,
    OUTPUT_MAX_LEN,
    MODEL_SAVE_PATH,
    SUFFIX,
)
from model import (
    TextRuleT5Model,
)
import torch

# OUTPUT_FILE = f"../cross_data/fstacred_model_trained_on_anchor_word_{str(RULE_TYPE)}_rules{SUFFIX}_for_nyt29.jsonl"
# OUTPUT_FILE = f"../combined/combined_model_trained_on_anchor_word_{str(RULE_TYPE)}_rules{SUFFIX}_for_nyt29.jsonl"
# OUTPUT_FILE = f"../combined/combined_model_trained_on_anchor_word_{str(RULE_TYPE)}_rules{SUFFIX}_for_fstacred.jsonl"
# PATH = "/workspace/fstacred/data_few_shot/_test_data_syntax_path.jsonl"

PATH = "/workspace/nyt29/_test_data_syntax_path.jsonl"
OUTPUT_FILE = f"../nyt29/model_gen_anchor_{str(RULE_TYPE)}_rules{SUFFIX}_bkgnd_pphrased.jsonl"

NUM_BEAMS = 4
syntax_rule_type = "org.clulab.odinsynth.evaluation.genericeval.SimplifiedSyntaxRuleType"
surface_rule_type = "org.clulab.odinsynth.evaluation.genericeval.SurfaceRuleType"
SCALA_RULE_TYPE_VALUE = syntax_rule_type if RULE_TYPE == RuleType.SYNTAX else surface_rule_type

def generate_rule(text, trained_model, tokenizer):
    inputs_encoding = tokenizer(
        text,
        max_length = INPUT_MAX_LEN,
        add_special_tokens = True,
        padding = 'max_length',
        truncation = True,
        return_attention_mask = True,
        return_tensors = "pt"
    )

    device = next(trained_model.parameters()).device

    inputs_encoding["input_ids"] = inputs_encoding["input_ids"].to(device)
    inputs_encoding["attention_mask"] = inputs_encoding["attention_mask"].to(device)

    generate_ids = trained_model.model.generate(
        input_ids = inputs_encoding["input_ids"],
        attention_mask = inputs_encoding["attention_mask"],
        max_length = OUTPUT_MAX_LEN,
        num_beams = NUM_BEAMS,
        num_return_sequences = 1,
        early_stopping = False,
    )

    preds = [
        tokenizer.decode(
            gen_id,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        for gen_id in generate_ids
    ]

    return preds[0]

torch.set_float32_matmul_precision('medium')
pl.seed_everything(100)
warnings.filterwarnings("ignore")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading data: ", PATH)
data = []
with open(PATH, "r") as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)

prepared_data = []
for i in range(len(data)):
    text, support = gen_input_text(
        data,
        i,
        get_rules = False,
        use_syntax_path_tokens = USE_SYNTAX,
        train = False,
    )
    prepared_data.append((text, support))

print("Loading model: ", MODEL_SAVE_PATH)
trained_model = TextRuleT5Model.load_from_checkpoint(MODEL_SAVE_PATH + '.ckpt')
trained_model.freeze()

print("Getting predictions")
results = []
for i in tqdm(range(len(prepared_data))):
    text = prepared_data[i][0]
    relation_name = prepared_data[i][1]["relation"]
    sentence = prepared_data[i][1]["sentence"]

    if (sentence["subjStart"] < sentence["objStart"]):
        direction = "org.clulab.odinsynth.evaluation.genericeval.SubjObjDirection"
    else:
        direction = "org.clulab.odinsynth.evaluation.genericeval.ObjSubjDirection"

    output = generate_rule(text, trained_model, tokenizer)
    if "No possible rule." in output:
        preds = []
    else:
        preds = output.split(" ~ ")

    if i%100 == 0:
        print(text, preds)

    patterns = []
    for pred in preds:
        pattern = {}
        pattern["pattern"] = pred
        pattern["relation"] = relation_name
        pattern["direction"] = {"$type": direction}
        pattern["weight"] = 1
        pattern["patternType"] = {"$type": SCALA_RULE_TYPE_VALUE}
        patterns.append(pattern)

    gen_rule = [prepared_data[i][1], patterns]
    results.append(gen_rule)

json_file = OUTPUT_FILE
with open(json_file, "w") as file:
    for result in results:
        file.write(json.dumps(result) + "\n")
print("Wrote to: ", json_file)
