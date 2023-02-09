# Transformer Distribution Visualizer

A simple Flask-based Web-UI application that allows users to input a prompt, select a Hugging Face transformer model, and specify a decoding strategy to generate text. The generated text, along with its word probabilities and top 10 next best words, can be displayed in an interactive fashion on the web page. 

## Getting Started

These instructions will help you set up the project on your local machine for development and testing purposes.

### Prerequisites

Before you begin, make sure you have the following installed:

- Python 3.6 or higher

- pip

### Installation

1. Clone the repository:

```
git clone https://github.com/aju22/transformer-model-web-ui.git
```

2. Navigate to the project directory:

```
cd transformer-model-web-ui
```

3. Create a virtual environment:

```
python -m venv env
```

4. Activate the virtual environment:

```
source env/bin/activate
```

5. Install the required packages:

```
pip install -r requirements.txt
```

### File Structure

- `models.py`: contains the `Transformer` class which is used to load and interact with the text generation model.

- `main.py`: the main file that runs the Flask app.

- `templates/index.html`: the template file that contains the HTML code for the app's homepage.

## Usage
To run the application, the following arguments can be passed as command line parameters:

```
python main.py --model <model_id> --tokenizer <tokenizer_id> --max_output_tokens <int> --gen_strat <str> --num_beams <int> --num_beam_groups <int> --penalty_alpha <float>
```

or simply, to run with default arguments:

```
python main.py
```

### Arguments
- `model` [STR, DEFAULT='gpt2']:  *The model id of a pretrained model hosted inside a model repo on huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.*

- `tokenizer` [STR, DEFAULT='gpt2']: *The model id of a predefined tokenizer hosted inside a model repo on huggingface.co. Valid model ids can be located at the root-level, like "bert-base-uncased", or namespaced under a user or organization name, like "bmdz/bert-base-german-cased".*

- `max_output_tokens` [INT, DEFAULT=100]: *The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.*

- `gen_strat` [str, DEFAULT='multinomial-sampling'], choices=['greedy', 'contrastive-search', 'multinomial-sampling', 'beam-search','beam-search-multinomial', 'diverse-beam-search']): *Specify output token sampling method.*

- `num_beams` [INT, DEFAULT=1]: *Number of beams for beam search. 1 means no beam search.*

- `num_beam_groups` [INT, DEFAULT=1]: *Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.*

- `penalty_alpha` [FLOAT, DEFAULT=0]: *The values balance the model confidence and the degeneration penalty in contrastive search decoding.*

#### *Note that `num_beams` should be divisible by `num_beam_groups` for group beam search.*


## Execution
When the app is run, it will load the specified model and tokenizer and start a local development server. Navigate to `http://localhost:5000/` in your web browser to interact with the app.

## Future Work

- Support for more decoding strategies.

- Option to load/change models and tokenzier on the webpage itself.

- Integration with more advanced visualization techniques to analyze the probability distributions of the transformer models.

## Contributing

Contributions are welcome to this project! If you have an idea for a feature or improvement, please open an issue or send a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE] file for details.



