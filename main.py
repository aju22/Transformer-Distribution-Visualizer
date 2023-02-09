from models import Transformer
from flask import Flask, request, render_template
import argparse



def load_model(model, tokenizer, max_output_tokens, gen_strat, num_beams, num_beam_groups, penalty_alpha):
    print("\nLoading Model.....\n")
    gen_model = Transformer(
        model_name=model,
        tokenizer_name=tokenizer,
        gen_strat=gen_strat,
        max_output_tokens=max_output_tokens,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        penalty_alpha=penalty_alpha
    )

    print('Model Loaded Successfully! \n')
    print(f"Using Model: {gen_model.model.__class__.__name__}")
    print(f"Using Tokenizer: {gen_model.tokenizer.__class__.__name__}")
    print(gen_model.desc, "\n")

    return gen_model


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    input_text = request.form['input_text']

    # call the processing function and store the result in a variable
    processed_json = process_function(input_text)

    # return the processed text as the response
    return processed_json


def process_function(input_text):
    global model
    # Your processing logic goes here
    processed_json = model.generate(input_text)
    return processed_json


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='gpt2', type=str, help="A string, the model id of a pretrained model hosted inside a model repo on huggingface.co. Valid model ids can be located at the root-level, like bert-base-uncased, or namespaced under a user or organization name, like dbmdz/bert-base-german-cased.")
    parser.add_argument("--tokenizer", default='gpt2', type=str, help="A string, the model id of a predefined tokenizer hosted inside a model repo on huggingface.co. Valid model ids can be located at the root-level, like bert-base-uncased, or namespaced under a user or organization name, like dbmdz/bert-base-german-cased.")
    parser.add_argument("--max_output_tokens", default=100, type=int, help="The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.")
    parser.add_argument("--gen_strat", default='multinomial-sampling', type=str, help="Specify output token sampling method.", choices=['greedy', 'contrastive-search', 'multinomial-sampling', 'beam-search','beam-search-multinomial', 'diverse-beam-search'])
    parser.add_argument("--num_beams", default=1, type=int, help="Number of beams for beam search. 1 means no beam search.")
    parser.add_argument("--num_beam_groups", default=1, type=int, help="Number of groups to divide num_beams into in order to ensure diversity among different groups of beams.")
    parser.add_argument("--penalty_alpha", default=0, type=float, help="The values balance the model confidence and the degeneration penalty in contrastive search decoding.")

    ARGS = parser.parse_args()

    if ARGS.num_beams % ARGS.num_beam_groups != 0:
        raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

    model = load_model(model=ARGS.model,
                       tokenizer=ARGS.tokenizer,
                       max_output_tokens=ARGS.max_output_tokens,
                       gen_strat=ARGS.gen_strat,
                       num_beams=ARGS.num_beams,
                       num_beam_groups=ARGS.num_beam_groups,
                       penalty_alpha=ARGS.penalty_alpha)

    app.run()

