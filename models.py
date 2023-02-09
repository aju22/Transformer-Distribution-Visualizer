from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from torch.nn.functional import softmax
import math
from torch import topk
import json
from collections import OrderedDict


class Transformer:

    def __init__(self, model_name, tokenizer_name, gen_strat, max_output_tokens, num_beams, num_beam_groups,
                 penalty_alpha):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, local_files_only=True)
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.gen_config = None
        self.desc=None
        self.max_output_tokens = max_output_tokens
        self.set_genConfig(gen_strat, num_beams, penalty_alpha, num_beam_groups)

    def set_genConfig(self, gen_strat, num_beams, penalty_alpha, num_beam_groups):

        if gen_strat == 'greedy':

            self.gen_config = GenerationConfig(max_new_tokens=self.max_output_tokens, do_sample=False, num_beams=1,
                                               output_scores=True,
                                               return_dict_in_generate=True,
                                               )

            self.desc = "Using Greedy Decoding."

        if gen_strat == 'contrastive-search':

            self.gen_config = GenerationConfig(max_new_tokens=self.max_output_tokens, do_sample=False, num_beams=1,
                                               penalty_alpha=penalty_alpha,
                                               top_k=50,
                                               output_scores=True,
                                               return_dict_in_generate=True,
                                               )

            self.desc = "Using Contrastive Search Decoding"

        if gen_strat == 'multinomial-sampling':

            self.gen_config = GenerationConfig(max_new_tokens=self.max_output_tokens, do_sample=True, num_beams=1,
                                               output_scores=True,
                                               return_dict_in_generate=True,
                                               )

            self.desc = "Using Multinomial Sampling"

        if gen_strat == 'beam-search':
            if num_beams <= 1:
                num_beams = 2

            self.gen_config = GenerationConfig(max_new_tokens=self.max_output_tokens, do_sample=False,
                                               num_beams=num_beams,
                                               output_scores=True,
                                               return_dict_in_generate=True,
                                               )

            self.desc = f"Using Beam-Search Decoding : NUM_BEAMS = {num_beams}"

        if gen_strat == 'beam-search-multinomial':

            if num_beams <= 1:
                num_beams = 2

            self.gen_config = GenerationConfig(max_new_tokens=self.max_output_tokens, do_sample=True,
                                               num_beams=num_beams,
                                               output_scores=True,
                                               return_dict_in_generate=True,
                                               )

            self.desc = f"Using Beam-Search-Multinomial Decoding : NUM_BEAMS = {num_beams}"

        if gen_strat == 'diverse-beam-search':

            if num_beams <= 1:
                num_beams = 2

            if num_beam_groups <= 1:
                num_beam_groups = 2

            self.gen_config = GenerationConfig(max_new_tokens=self.max_output_tokens, do_sample=False,
                                               num_beams=num_beams,
                                               num_beam_groups=num_beam_groups,
                                               output_scores=True,
                                               return_dict_in_generate=True,
                                               )
            self.desc = f"Using Diverse-Beam-Search Decoding : NUM_BEAMS = {num_beams} NUM_BEAM_GROUPS = {num_beam_groups}"

    def compute_scores(self, outputs):
        seq_scores = []

        for i in range(len(outputs.scores)):
            res = topk(softmax(outputs.scores[i], dim=1), 10, dim=1)
            top_k_dict = OrderedDict()

            for j in range(10):

                word = self.tokenizer.decode(res[1][0][j], skip_special_tokens=True)

                if word == '\n':
                    word = "&lt;endl&gt;"

                if word == "<|endoftext|>":
                    word = "&lt;|endoftext|&gt;"

                top_k_dict[word] = res[0][0][j].item()

            seq_scores.append(top_k_dict)

        return seq_scores

    def generate(self, prompt: str):

        inputs = self.tokenizer([prompt], return_tensors="pt")
        outputs = self.model.generate(**inputs, generation_config=self.gen_config)

        seq_score = self.compute_scores(outputs)

        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        gen_tokens = outputs.sequences
        text = []
        for token in gen_tokens[0]:

            word = self.tokenizer.decode(token, skip_special_tokens=True)

            if word == '\n':
                word = "&lt;endl&gt;"

            if word == "<|endoftext|>":
                word = "&lt;|endoftext|&gt;"

            text.append(word)

        info_json = {}
        input_length = inputs.input_ids.shape[1]

        for i in range(len(text)):
            if i < input_length:
                word_string = str(text[i])
                score = -1
                info_json[i] = {word_string: score}
            else:
                if str(text[i]) in seq_score[i - input_length]:
                    seq_score[i - input_length].move_to_end(str(text[i]))
                else:
                    seq_score[i - input_length][str(text[i])] = math.exp(transition_scores[0][i - input_length].item())

                info_json[i] = seq_score[i - input_length]

        json_object = json.dumps(info_json, indent=4)

        return json_object
