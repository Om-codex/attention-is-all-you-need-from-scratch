import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class TranslationDataset(Dataset):

    def __init__(self, english_sentence, french_sentence, tokenizer, max_len = 50):
        self.eng = english_sentence
        self.fr = french_sentence
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.sos_id = tokenizer.sp.bos_id()
        self.eos_id = tokenizer.sp.eos_id()
        self.pad_id = tokenizer.sp.pad_id()

    def __len__(self):
        return len(self.eng)
    
    def __getitem__(self, idx):

        # get sentence pair
        eng_sentence = self.eng[idx]
        fr_sentence = self.fr[idx]

        # tokenize
        encoder_tokens = self.tokenizer.encode(eng_sentence)
        decoder_tokens = self.tokenizer.encode(fr_sentence)

        # truncate if longer than max_len
        encoder_tokens = encoder_tokens[:self.max_len - 1]
        decoder_tokens = decoder_tokens[:self.max_len - 1]

        # Encoder Inputs
        encoder_inputs = encoder_tokens + [self.eos_id]

        # Decoder Inputs
        decoder_inputs =[self.sos_id]+ decoder_tokens

        # Target output
        target = decoder_tokens + [self.eos_id]

        return {
            "encoder_input" : torch.tensor(encoder_inputs, dtype = torch.long),
            "decoder_input" : torch.tensor(decoder_inputs, dtype = torch.long),
            "target" : torch.tensor(target, dtype = torch.long)
        }
    
    def collate_fn(self, batch):
        encoder_inputs = [item["encoder_input"] for item in batch]
        decoder_inputs = [item["decoder_input"] for item in batch]
        targets = [item["target"] for item in batch]

        encoder_inputs = pad_sequence(
            encoder_inputs,
            batch_first = True,
            padding_value = self.pad_id
        )

        decoder_inputs = pad_sequence(
            decoder_inputs,
            batch_first = True,
            padding_value = self.pad_id
        )

        targets = pad_sequence(
            targets,
            batch_first = True,
            padding_value = self.pad_id
        )

        return {
            "encoder_inputs" : encoder_inputs,
            "decoder_inputs" : decoder_inputs,
            "targets" : targets
        }
